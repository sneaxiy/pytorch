#include <torch/csrc/distributed/autograd/context/thread_local_context.h>

namespace torch {
namespace distributed {
namespace autograd {

namespace {
thread_local ContextPtr::weak_type tl_context_weak_ptr;
} // namespace

ThreadLocalDistAutogradContext::ThreadLocalDistAutogradContext()
    : prev_context_weak_ptr_(tl_context_weak_ptr),
      context_weak_ptr_(tl_context_weak_ptr) {}

ThreadLocalDistAutogradContext::ThreadLocalDistAutogradContext(
    ContextPtr::weak_type&& context_wp)
    : prev_context_weak_ptr_(tl_context_weak_ptr),
      context_weak_ptr_(std::move(context_wp)) {
  tl_context_weak_ptr = context_weak_ptr_;
}

ThreadLocalDistAutogradContext::~ThreadLocalDistAutogradContext() {
  tl_context_weak_ptr = prev_context_weak_ptr_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
