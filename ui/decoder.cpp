#include "decoder.h"

#include "buffer.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include <stb_image.h>

namespace cortex {

namespace {

class decoder_impl;

class frame_impl final : public frame
{
  friend decoder_impl;

  std::vector<uint8_t> data_;

  size_t width_{};

  size_t height_{};

public:
  auto data() const -> const uint8_t* override { return data_.data(); }

  auto width() const -> size_t override { return width_; }

  auto height() const -> size_t override { return height_; }
};

class decoder_impl final : public decoder
{
  void* callback_data_{};
  Callback callback_{};

  std::queue<std::unique_ptr<buffer>> job_queue_;
  std::queue<std::unique_ptr<frame_impl>> result_queue_;

  std::thread worker_;
  std::mutex job_mutex_;
  std::mutex result_mutex_;
  std::condition_variable job_cv_;
  std::atomic<bool> running_{ true };

public:
  decoder_impl()
  {
    worker_ = std::thread([this]() {
      while (running_) {
        std::unique_ptr<buffer> job;
        {
          std::unique_lock lock(job_mutex_);
          job_cv_.wait(lock, [this]() { return !job_queue_.empty() || !running_; });
          if (!running_ && job_queue_.empty())
            break;
          job = std::move(job_queue_.front());
          job_queue_.pop();
        }
        process_job(std::move(job));
      }
    });
  }

  ~decoder_impl() override { terminate(); }

  void setup(void* callback_data, Callback callback) override
  {
    callback_data_ = callback_data;
    callback_ = callback;
  }

  void terminate() override
  {
    running_ = false;
    job_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void loop() override
  {
    std::unique_ptr<frame_impl> frame;
    {
      std::lock_guard lock(result_mutex_);
      if (!result_queue_.empty()) {
        frame = std::move(result_queue_.front());
        result_queue_.pop();
      }
    }
    if (frame && callback_) {
      callback_(callback_data_, std::move(frame));
    }
  }

  void queue(std::unique_ptr<buffer> buffer) override
  {
    {
      std::lock_guard lock(job_mutex_);
      job_queue_.emplace(std::move(buffer));
    }
    job_cv_.notify_one();
  }

  [[nodiscard]] auto busy() -> bool override
  {
    std::lock_guard lock(job_mutex_);
    return !job_queue_.empty();
  }

private:
  void process_job(std::unique_ptr<buffer> buf)
  {
    const auto* raw_data = buf->data();

    // assume raspberry pi camera module 2
    constexpr auto w{ 3820 };
    constexpr auto h{ 2464 };
    constexpr auto num_pixels{ w * h };

    auto frame = std::make_unique<frame_impl>();
    frame->data_.resize(w * h);
    frame->width_ = w;
    frame->height_ = h;

    for (size_t i = 0; i < num_pixels; i++) {
    }

    {
      std::lock_guard lock(result_mutex_);
      result_queue_.emplace(std::move(frame));
    }
  }
};

} // namespace

auto
decoder::create() -> std::unique_ptr<decoder>
{
  return std::make_unique<decoder_impl>();
}

} // namespace cortex
