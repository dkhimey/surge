// thread_pool.h
#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop(false) {
        workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this]() { this->worker_loop(); });
        }
    }

    // Enqueue a task (returns a future to get the result if needed)
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> 
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop.load()) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        cond.notify_one();
        return res;
    }

    // Graceful shutdown: waits for tasks to complete
    void shutdown() {
        stop.store(true);
        cond.notify_all();
        for (auto &w : workers) {
            if (w.joinable()) w.join();
        }
    }

    ~ThreadPool() {
        if (!stop.load()) {
            shutdown();
        }
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cond.wait(lock, [this]() { return stop.load() || !tasks.empty(); });
                if (stop.load() && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable cond;
    std::atomic<bool> stop;
};
