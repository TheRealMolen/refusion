#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <numbers>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>

#include "pixie.h"

#define D_USE_WORKERS

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
        Cell operator-(const Cell& o) const
        {
            return { a - o.a, b - o.b };
        }
        Cell& operator+=(const Cell& o)
        {
            a += o.a;
            b += o.b;
            return *this;
        }
        Cell& operator-=(const Cell& o)
        {
            a -= o.a;
            b -= o.b;
            return *this;
        }
    };

    size_t width, height;
    vector<Cell> cells;
    vector<Cell> next;
    vector<Cell> published;

    // sim params
    float diffusionA = 0.6f;
    float diffusionB = 0.2f;
    float feedRate = 0.055f;//0.055f;
    float killRate = 0.062f;//0.062f;
    float feedKillAngle = 0.0f;

    
    Sim(size_t w, size_t h)
        : width(w), height(h), cells(width*height, { 1.0f, 0.0f })
    {
        // seed a little B
        uint32_t seedSize = 100;//uint32_t(width);
        for (uint32_t y = 0; y < seedSize; ++y)
        {
            for (uint32_t x = 0; x < seedSize; ++x)
            {
                if (x*x + y*y > seedSize* seedSize)
                    continue;

                cells[x + y * width].a = 0.0f;
                cells[x + y * width].b = 1.0f;
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


#ifdef D_USE_WORKERS
    static const int WorkerPoolSize = 10;

    unique_ptr<std::thread> simThread;
    mutable std::mutex publishCs;
    vector<std::thread> workers;
    std::atomic_bool shouldQuit{ false };
    struct alignas(64) WorkerCommand
    {
        std::atomic_uint64_t val{0};
        double avgTime = 0;
        double numTicks = 0;
    };
    WorkerCommand workerBusy[WorkerPoolSize];

    void workerFunc(int ix, uint32_t startY, uint32_t endY);
    void tickSubset(uint32_t startY, uint32_t endY);
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

        auto& worker = workers.emplace_back(&Sim::workerFunc, this, i, startY, endY);
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
    const float twoPi = std::numbers::pi_v<float> * 2.0f;

    auto start = std::chrono::high_resolution_clock::now();
    double secsSinceLastWrap = 0.0;

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

        if (publishCs.try_lock())
        {
            published.swap(next);
            publishCs.unlock();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        start = end;
        secsSinceLastWrap += dur.count();

        static const double wrapTimeSecs = 10.0;
        feedKillAngle += float((dur.count() * twoPi / wrapTimeSecs));
        if (feedKillAngle > twoPi)
        {
            feedKillAngle -= twoPi;
            secsSinceLastWrap = 0.0;
        }

        static constexpr float feedOrg = 0.032f;
        static constexpr float feedScl = 0.002f;
        static constexpr float killOrg = 0.060f;
        static constexpr float killScl = 0.002f;
        feedRate = cosf(feedKillAngle) * feedScl + feedOrg;
        killRate = sinf(feedKillAngle) * killScl + killOrg;
    }
}

void Sim::workerFunc(int ix, uint32_t startY, uint32_t endY)
{
    for (;;)
    {
        workerBusy[ix].val.wait(0);
        uint64_t command = workerBusy[ix].val.load();
        if (command == 0)
            continue;
        if (command > 1)
            return;

        auto start = std::chrono::high_resolution_clock::now();

        tickSubset(startY, endY);

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

void Sim::tickSubset(uint32_t startY, uint32_t endY)
{
    auto out = next.data() + (width * startY);
    auto in = cells.data() + (width * startY);

    const ptrdiff_t ul = -1 - width;
    const ptrdiff_t up = 0 - width;
    const ptrdiff_t ur = 1 - width;
    const ptrdiff_t lf = -1;
    const ptrdiff_t rt = 1;
    const ptrdiff_t dl = width - 1;
    const ptrdiff_t dn = width;
    const ptrdiff_t dr = 1 + width;

    float recipWidth = 1.0f / float(width);
    float recipHeight = 1.0f / float(height);

    for (size_t y = startY; y < endY; ++y)
    {
        size_t py = (y > 0) ? (y - 1) : y;
        size_t ny = (y + 1 < height) ? (y + 1) : y;

        float feed = 0.1f - 0.09f * float(y) * recipHeight;
       // feed = feedRate;

        for (size_t x = 0; x < width; ++x, ++out, ++in)
        {
            // laplacian based on https://www.karlsims.com/rd.html
            Cell laplacian;
            if (x > 0 && y > 0 && x + 1 < width && y + 1 < height)
            {
                laplacian =
                    0.2f * (*(in + up) + *(in + lf) + *(in + rt) + *(in + dn)) +
                    + 0.05f * (*(in + ul) + *(in + ur) + *(in + dl) + *(in + dr))
                    - *in;
            }
            else
            {
                size_t px = (x > 0) ? (x - 1) : x;
                size_t nx = (x + 1 < width) ? (x + 1) : x;

                laplacian =  0.2f * (cells[py * width + x] + cells[y * width + px] + cells[y * width + nx] + cells[ny * width + x]);
                laplacian += 0.05f * (cells[py * width + px] + cells[py * width + nx] + cells[ny * width + px] + cells[ny * width + nx]);
                laplacian -= *in;
            }

            float kill = 0.045f + 0.025f * float(x) * recipWidth;
          //  kill = killRate;

            float ab2 = in->a * in->b * in->b;
            out->a = in->a + (diffusionA * laplacian.a - ab2 + feed * (1.0f - in->a));
            out->b = in->b + (diffusionB * laplacian.b + ab2 - (kill + feed) * in->b);
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
    const size_t dl = width - 1;
    const size_t dn = width;
    const size_t dr = 1 + width;

    float recipWidth = 1.0f / float(width);
    float recipHeight = 1.0f / float(height);

    for (size_t y = 0; y < height; ++y)
    {
        float feed = 0.01f + 0.09f * float(y) * recipHeight;
        _ASSERT(feed >= 0.0096 && feed <= 0.102);
        _ASSERT(isfinite(feed));

        feed = feedRate;

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

            float kill = 0.045f + 0.025f * float(x) * recipWidth;
            _ASSERT(kill >= 0.043 && kill <= 0.072);
            _ASSERT(isfinite(kill));
            kill = killRate;

            float ab2 = in->a * in->b * in->b;
            _ASSERT(isfinite(ab2));
            out->a = in->a + (diffusionA * laplacian.a - ab2 + feed * (1.0f - in->a));// * deltatime;
            _ASSERT(out->a >= 0.0f);
            _ASSERT(out->a <= 1.0f);
            _ASSERT(isfinite(out->a));
            _ASSERT(fabsf(out->a) < 100.0f);
            out->b = in->b + (diffusionB * laplacian.b + ab2 - (kill + feed) * in->b);// * deltatime;
            _ASSERT(out->b >= 0.0f);
            _ASSERT(out->b <= 1.0f);
            _ASSERT(isfinite(out->b));
            _ASSERT(fabsf(out->b) < 100.0f);
        }
    }

    cells.swap(next);
    std::ranges::copy(cells, begin(published));
#else
    // this all happens inside simThreadFunc
#endif
}

__declspec(noinline) void Sim::render(uint32_t* pixels, size_t w, size_t h) const
{
    _ASSERT(w == width && h == height);
    _ASSERT(pixels);

#ifdef D_USE_WORKERS
    ScopeCs cs(publishCs);
#endif

    const Cell* cell = published.data();
    uint32_t* pixel = pixels;
    for ( ; pixel != pixels + (width * height); ++pixel, ++cell)
    {
        uint32_t col = 0xff88ff;
        //if (cell->b > 0.0f)
        //{
        //    //col = 0;

        //    float concA = cell->a / (cell->a + cell->b);
        //    concA *= concA;
        //    concA *= concA;
        //    int i = clamp(int(concA * 255.0f), 0, 0xff);
        //    col = uint32_t((i<<16) | (i<<8) | i);
        //}
        //else if (cell->a > 0.01f)
        //{
        //    col = 0xffffff;
        //}

        float bias = cell->a - cell->b;
        bias -= 0.2f;
        bias *= 10.0f;
        int i = clamp(int(bias * 255.0f), 40, 0xff);
        col = uint32_t((i<<16) | (i<<8) | i);

        *pixel = col;
    }
}



int main()
{
    int width = 800;
    int height = 500;

    Pixie::Window win;
    const bool fullscreen = false;
    if (!win.Open("refusion", width, height, fullscreen))
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
