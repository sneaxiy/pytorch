#pragma once

#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch {
namespace distributed {
namespace autograd {

class ThreadLocalDistAutogradContext {
 public:
  ThreadLocalDistAutogradContext();
  explicit ThreadLocalDistAutogradContext(ContextPtr::weak_type&& context_wp);
  ~ThreadLocalDistAutogradContext();

  ContextPtr::weak_type context_weak_ptr_;

 private:
  ContextPtr::weak_type prev_context_weak_ptr_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
