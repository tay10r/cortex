#include "api.h"

#include "wbc_differential_api.h"

#include <nlohmann/json.hpp>

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace cortex {

namespace {

using api_factory = auto (*)() -> std::unique_ptr<api>;

struct api_entry final
{
  const char* name;

  api_factory factory;
};

const std::array<api_entry, 1> apis = { { "wbc_differential", []() -> std::unique_ptr<api> {
                                           return std::make_unique<wbc_differential_api>();
                                         } } };

class async_setup_cmd;
class async_teardown_cmd;
class async_reset_cmd;
class async_update_cmd;
class async_finalize_cmd;

class async_cmd_visitor
{
public:
  virtual ~async_cmd_visitor() = default;

  virtual void visit(async_setup_cmd&) = 0;

  virtual void visit(async_teardown_cmd&) = 0;

  virtual void visit(async_reset_cmd&) = 0;

  virtual void visit(async_update_cmd&) = 0;

  virtual void visit(async_finalize_cmd&) = 0;
};

class async_cmd
{
public:
  virtual ~async_cmd() = default;

  virtual void accept(async_cmd_visitor&) = 0;
};

template<typename Derived>
class async_cmd_base : public async_cmd
{
public:
  void accept(async_cmd_visitor& v) override { v.visit(static_cast<Derived&>(*this)); }
};

class async_setup_cmd final : public async_cmd_base<async_setup_cmd>
{
public:
};

class async_teardown_cmd final : public async_cmd_base<async_teardown_cmd>
{
public:
};

class async_reset_cmd final : public async_cmd_base<async_reset_cmd>
{
  report_header header_;

public:
  async_reset_cmd(const report_header& header)
    : header_(header)
  {
  }

  [[nodiscard]] auto header() const -> const report_header& { return header_; }
};

class async_update_cmd final : public async_cmd_base<async_update_cmd>
{
  std::string encoded_img_;

public:
  async_update_cmd(std::string encoded_img)
    : encoded_img_(std::move(encoded_img))
  {
  }

  [[nodiscard]] auto take() -> std::string { return std::move(encoded_img_); }
};

class async_finalize_cmd final : public async_cmd_base<async_finalize_cmd>
{
public:
};

class async_api final
  : public api
  , public async_cmd_visitor
{
  std::string inner_api_name_;

  std::unique_ptr<api> inner_api_;

  std::thread thread_;

  std::mutex cmds_lock_;

  const size_t max_queue_size_{};

  std::vector<std::unique_ptr<async_cmd>> cmds_;

  std::condition_variable cmds_cv_;

  std::mutex results_lock_;

  nlohmann::json results_;

  bool stop_flag_{ false };

public:
  explicit async_api(std::string inner_api, const size_t max_queue_size)
    : inner_api_name_(std::move(inner_api))
    , max_queue_size_(max_queue_size)
  {
    thread_ = std::thread(&async_api::run_thread, this);
  }

  ~async_api()
  {
    // in case the API was not shutdown explicity
    {
      std::lock_guard<std::mutex> lock(cmds_lock_);
      stop_flag_ = true;
    }

    cmds_cv_.notify_one();

    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void setup() override { push_cmd(std::make_unique<async_setup_cmd>()); }

  void teardown() override
  {
    push_cmd(std::make_unique<async_teardown_cmd>());

    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void reset(const report_header& header) override { push_cmd(std::make_unique<async_reset_cmd>(header)); }

  void update(std::string encoded_img) override
  {
    push_cmd(std::make_unique<async_update_cmd>(std::move(encoded_img)));
  }

  void finalize() override { push_cmd(std::make_unique<async_finalize_cmd>()); }

  auto results() -> nlohmann::json override
  {
    nlohmann::json results;
    {
      std::lock_guard<std::mutex> lock(results_lock_);
      results = results_;
    }
    return results;
  }

protected:
  void push_cmd(std::unique_ptr<async_cmd> cmd)
  {
    auto max_exceeded{ false };

    {
      std::lock_guard<std::mutex> lock(cmds_lock_);
      if (cmds_.size() < max_queue_size_) {
        cmds_.emplace_back(std::move(cmd));
      } else {
        max_exceeded = true;
      }
    }

    if (max_exceeded) {
      throw std::runtime_error("Max queue size exceeded.");
    }

    cmds_cv_.notify_one();
  }

  void run_thread()
  {
    inner_api_ = api::create(inner_api_name_);

    while (!stop_flag_) {

      std::unique_ptr<async_cmd> cmd;

      {
        std::unique_lock<std::mutex> lock(cmds_lock_);

        cmds_cv_.wait(lock, [this] { return stop_flag_ || !cmds_.empty(); });

        if (stop_flag_ && cmds_.empty()) {
          // No more commands to process; we can exit
          break;
        }

        if (!cmds_.empty()) {
          cmd = std::move(cmds_.front());
          cmds_.erase(cmds_.begin());
        }
      }

      if (cmd) {
        // Execute the command via visitor
        cmd->accept(*this);

        // After each command, we can update results_
        // so the user sees the latest after every operation
        {
          std::lock_guard<std::mutex> rlock(results_lock_);
          results_ = inner_api_->results();
        }
      }
    }
  }

  void visit(async_setup_cmd&) override { inner_api_->setup(); }

  void visit(async_teardown_cmd&) override
  {
    inner_api_->teardown();
    stop_flag_ = true;
  }

  void visit(async_reset_cmd& cmd) override { inner_api_->reset(cmd.header()); }

  void visit(async_update_cmd& cmd) override { inner_api_->update(cmd.take()); }

  void visit(async_finalize_cmd& cmd) override { return inner_api_->finalize(); }
};

} // namespace

auto
api::create(const std::string& name) -> std::unique_ptr<api>
{
  for (const auto& entry : apis) {
    if (name == entry.name) {
      return entry.factory();
    }
  }

  std::ostringstream stream;
  stream << "unknown api '" << name << "'";
  throw std::runtime_error(stream.str());
}

auto
api::create_async(const std::string& name, const size_t max_queue_size) -> std::unique_ptr<api>
{
  return std::make_unique<async_api>(name, max_queue_size);
}

} // namespace cortex
