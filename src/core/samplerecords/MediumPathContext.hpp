#pragma once
#include <memory>
namespace Tungsten {
// Persistent branch-local state. Reset replaces a pointer; it never mutates a
// context shared with a continuation, a light sample or another shadow branch.
struct MediumPathContext { virtual ~MediumPathContext() = default; };
struct MediumEventContext { virtual ~MediumEventContext() = default; };
}
