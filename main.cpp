#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>

#include "pixie.h"

#define D_USE_WORKERS

using namespace std::chrono_literals;
using std::unique_ptr;
using std::vector;
using ScopeCs = std::lock_guard<std::mutex>;
using std::begin;
using std::end;

vector<std::wstring> g_staticStrings;

struct Sim
{
    struct Cell
    {
        float a, b;

        Cell operator+(const Cell& o) const
        {
            return { a + o.a, b + o.b };
        }
        Cell& operator+=(const Cell& o)
        {
            a += o.a;
            b += o.b;
            return *this;
        }
    };

    size_t width, height;
    vector<Cell> cells;
    vector<Cell> next;
    vector<Cell> published;

    // sim params
    float diffusionA = 1.0f;
    float diffusionB = 0.5f;
    float feedRate = 0.056f;//0.055f;
    float killRate = 0.064f;//0.062f;

    
    Sim(size_t w, size_t h)
        : width(w), height(h), cells(width*height, { 1.0f, 0.0f })
    {
        // seed a little B
        uint32_t seedSize = 100;
        for (uint32_t y = 0; y < seedSize; ++y)
        {
            for (uint32_t x = 0; x < seedSize; ++x)
            {
                if (x*x + y*y > seedSize* seedSize)
                    continue;

                cells[x + y*width].b = 1.0f;
            }
        }

        next.resize(cells.size());
        published.resize(cells.size());
        std::ranges::copy(cells, begin(published));

        initWorkers();
    }

    ~Sim()
    {
        teardownWorkers();
    }

    void tick(float deltatime);
    void render(uint32_t* pixels, size_t w, size_t h) const;


    // ----- workers ------
#ifdef D_USE_WORKERS
    static const int WorkerPoolSize = 10;
    //struct WorkerCommand
    //{
    //    uint32_t startY = ~0u;       // if ==~0u means "quit"
    //    uint32_t endY;
    //    float    deltatime;
    //};

    unique_ptr<std::thread> simThread;
    //std::mutex workQueueCs;
    mutable std::mutex publishCs;
    vector<std::thread> workers;
    //vector<WorkerCommand> commands;
    std::atomic_bool shouldQuit{ false };
    //std::atomic<int64_t> incompleteCommands;
    struct alignas(64) WorkerCommand
    {
        std::atomic_uint64_t val{0};
        double avgTime = 0;
        double numTicks = 0;
    };
    WorkerCommand workerBusy[WorkerPoolSize];

    void workerFunc(int ix, uint32_t startY, uint32_t endY, float deltaTime);
    void tickSubset(uint32_t startY, uint32_t endY, float deltaTime);
    void simThreadFunc();
#endif

    void initWorkers();
    void teardownWorkers();
};

inline Sim::Cell operator*(float k, const Sim::Cell& c)
{
    return { k * c.a, k * c.b };
}


void Sim::initWorkers()
{
#ifdef D_USE_WORKERS
    static_assert(sizeof workerBusy == 64 * WorkerPoolSize);

    workers.reserve(WorkerPoolSize);

    const float deltaTime = 1.0f;

    std::cout << "setting up workers...\n";

    uint32_t rowsPerWorker = uint32_t(height / WorkerPoolSize);
    uint32_t startY = 0;
    uint32_t endY = startY + rowsPerWorker;
    for (int i = 0; i < WorkerPoolSize; ++i)
    {
        if (i + 1 == WorkerPoolSize)
            endY = uint32_t(height);

        std::cout << "   worker " << i << " responsible for " << startY << " to " << endY << " : " << (endY-startY) << " rows\n";

        workerBusy[i].val = 0;

        auto& worker = workers.emplace_back(&Sim::workerFunc, this, i, startY, endY, deltaTime);
        auto& name = g_staticStrings.emplace_back(std::format(L"Worker {}" , i));
        HANDLE threadHandle = HANDLE(worker.native_handle());
        SetThreadDescription(threadHandle, name.c_str());

        startY = endY;
        endY += rowsPerWorker;
    }

    simThread = std::make_unique<std::thread>(&Sim::simThreadFunc, this);
    HANDLE threadHandle = HANDLE(simThread->native_handle());
    SetThreadDescription(threadHandle, L"Sim Thread");
#endif
}

void Sim::teardownWorkers()
{
#ifdef D_USE_WORKERS
    /*{
        std::lock_guard<std::mutex> lock(workQueueCs);
        for (auto& worker : workers)
            commands.emplace_back();
        incompleteCommands += int64_t(workers.size());
    }*/
    //shouldQuit = true;

    shouldQuit = true;
    if (simThread.get())
        simThread->join();

    for (auto& command : workerBusy)
    {
        while (command.val.exchange(100) != 100)
        { /**/ }
        command.val.notify_one();
    }

    for (auto& worker : workers)
        worker.join();

    std::cout << "timing:\n";
    for (int i=0; i<WorkerPoolSize; ++i)
    {
        std::cout << "  worker" << i << ": " << (workerBusy[i].avgTime*1000.0) << "ms\n";
    }
#endif
}


#ifdef D_USE_WORKERS
void Sim::simThreadFunc()
{
    while (!shouldQuit)
    {
        _ASSERT(!std::any_of(begin(workerBusy), end(workerBusy), [](const auto& command) { return command.val.load() > 0; }));
        for (auto& command : workerBusy)
        {
            command.val.store(1);
            command.val.notify_one();
        }

        while (std::any_of(begin(workerBusy), end(workerBusy), [](const auto& command) { return command.val.load() > 0; }))
        {
            std::this_thread::yield();

            if (shouldQuit)
                return;
        }

        cells.swap(next);

        {
            if (!publishCs.try_lock())
                continue;

            //std::ranges::copy(cells, begin(published));
            published.swap(next);

            publishCs.unlock();
        }
    }
}

void Sim::workerFunc(int ix, uint32_t startY, uint32_t endY, float deltaTime)
{
    for (;;)
    {
        //if (shouldQuit.load())
        //    return;

        /*
        if (incompleteCommands.load() == 0)
        {
            std::this_thread::yield();
            continue;
        }*/
        workerBusy[ix].val.wait(0);
        uint64_t command = workerBusy[ix].val.load();
        if (command == 0)
        {
            //std::this_thread::yield();
            continue;
        }
        if (command > 1)
            return;

        auto start = std::chrono::high_resolution_clock::now();

        /*
        WorkerCommand command;
        {
            std::lock_guard<std::mutex> lock(workQueueCs);
            if (commands.empty())
                continue;

            command = commands.back();
            commands.pop_back();
        }*/

        tickSubset(startY, endY, deltaTime);

        //--incompleteCommands;
        if (!workerBusy[ix].val.compare_exchange_strong(command, 0))
            break;

        // timing....
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        ++workerBusy[ix].numTicks;
        double oldAvg = workerBusy[ix].avgTime;
        double newAvg = oldAvg + ((dur.count() - oldAvg) / double(workerBusy[ix].numTicks));
        workerBusy[ix].avgTime = newAvg;
    }
}

void Sim::tickSubset(uint32_t startY, uint32_t endY, float deltaTime)
{
    auto out = next.data() + (width * startY);
    auto in = cells.data() + (width * startY);

    const size_t ul = -1 - width;
    const size_t up = 0 - width;
    const size_t ur = 1 - width;
    const size_t lf = -1;
    const size_t rt = 1;
    const size_t dl = 1 + width;
    const size_t dn = width;
    const size_t dr = 1 + width;
    for (size_t y = startY; y < endY; ++y)
    {
        for (size_t x = 0; x < width; ++x, ++out, ++in)
        {
            // laplacian based on https://www.karlsims.com/rd.html
            Cell laplacian;
            if (x > 0 && y > 0 && x + 1 < width && y + 1 < height)
            {
                laplacian = -1.0f * *in +
                    0.2f * (*(in + up) + *(in + lf) + *(in + rt) + *(in + dn)) +
                    0.05f * (*(in + ul) + *(in + ur) + *(in + dl) + *(in + dr));
            }
            else
            {
                size_t px = (x > 0) ? (x - 1) : (width - 1);
                size_t nx = (x + 1 < width) ? (x + 1) : 0;
                size_t py = (y > 0) ? (y - 1) : (height - 1);
                size_t ny = (y + 1 < height) ? (y + 1) : 0;

                laplacian = -1.0f * *in;
                laplacian += 0.2f * (cells[py * width + x] + cells[y * width + px] + cells[y * width + nx] + cells[ny * width + x]);
                laplacian += 0.05f * (cells[py * width + px] + cells[py * width + nx] + cells[ny * width + px] + cells[ny * width + nx]);
            }

            float ab2 = in->a * in->b * in->b;
            out->a = in->a + (diffusionA * laplacian.a - ab2 + feedRate * (1.0f - in->a)) * deltaTime;
            out->b = in->b + (diffusionB * laplacian.b + ab2 - (killRate + feedRate) * in->b) * deltaTime;
        }
    }
}
#endif



void Sim::tick(float deltatime)
{
#ifndef D_USE_WORKERS
    auto out = std::begin(next);
    auto in = std::begin(cells);

    const size_t ul = -1 - width;
    const size_t up = 0 - width;
    const size_t ur = 1 - width;
    const size_t lf = -1;
    const size_t rt = 1;
    const size_t dl = 1 + width;
    const size_t dn = width;
    const size_t dr = 1 + width;
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x, ++out, ++in)
        {
            // laplacian based on https://www.karlsims.com/rd.html
            Cell laplacian;
            if (x>0 && y>0 && x+1<width && y+1<height)
            {
                laplacian = -1.0f * *in +
                            0.2f * (*(in + up) + *(in + lf) + *(in + rt) + *(in + dn)) +
                            0.05f * (*(in + ul) + *(in + ur) + *(in + dl) + *(in + dr));
            }
            else
            {
                size_t px = (x > 0) ? (x - 1) : (width - 1);
                size_t nx = (x + 1 < width) ? (x + 1) : 0;
                size_t py = (y > 0) ? (y - 1) : (height- 1);
                size_t ny = (y + 1 < height) ? (y + 1) : 0;

                laplacian = -1.0f * *in;
                laplacian += 0.2f * (cells[py * width + x] + cells[y * width + px] + cells[y * width + nx] + cells[ny * width + x]);
                laplacian += 0.05f * (cells[py * width + px] + cells[py * width + nx] + cells[ny * width + px] + cells[ny * width + nx]);
            }

            float ab2 = in->a * in->b * in->b;
            out->a = in->a + (diffusionA * laplacian.a - ab2 + feedRate * (1.0f - in->a)) * deltatime;
            out->b = in->b + (diffusionB * laplacian.b + ab2 - (killRate + feedRate) * in->b) * deltatime;
        }
    }

    cells.swap(next);
#else
    //_ASSERT(incompleteCommands == 0);
    //uint32_t rowsPerWorker = uint32_t(height / WorkerPoolSize);
    {
        /*
        std::lock_guard<std::mutex> cs(workQueueCs);
        uint32_t startY = 0;
        uint32_t endY = startY + rowsPerWorker;
        for (int i = 0; i<WorkerPoolSize; ++i)
        {
            if (i+1 == WorkerPoolSize)
                endY = uint32_t(height);

            commands.emplace_back(startY, endY, deltatime);

            startY = endY;
            endY += rowsPerWorker;
        }
        */

        //incompleteCommands += WorkerPoolSize;
    }
#endif
}

__declspec(noinline) void Sim::render(uint32_t* pixels, size_t w, size_t h) const
{
    _ASSERT(w == width && h == height);
    _ASSERT(pixels);

    ScopeCs cs(publishCs);

    const Cell* cell = published.data();
    uint32_t* pixel = pixels;
    for ( ; pixel != pixels + (width * height); ++pixel, ++cell)
    {
        uint32_t col = 0xff88ff;
        if (cell->b > 0.001f)
        {
            float concA = cell->a / (cell->a + cell->b);
            concA *= concA;
            concA *= concA;
            int i = clamp(int(concA * 255.0f), 0, 0xff);
            col = uint32_t((i<<16) | (i<<8) | i);
        }
        else if (cell->a > 0.0f)
        {
            col = 0xffffff;
        }

        *pixel = col;
    }
}



int main()
{
    int width = 400;
    int height = 400;

    Pixie::Window win;
    const bool fullscreen = false;
    if (!win.Open("Reaction/Diffusion", width, height, fullscreen))
        return 1;

    Sim sim(width, height);

    while (!win.HasKeyGoneUp(Pixie::Key_Escape))
    {
        uint32_t* pixels = win.GetPixels();
        sim.render(pixels, width, height);
        if (!win.Update())
            return 2;

        sim.tick(1.0f);
    }

    return 0;
}
