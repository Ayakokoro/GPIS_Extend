#include "cuda/RaySegmentBatch.hpp"
#include "cuda/SeSplat1D.hpp"

#include <cmath>
#include <limits>

namespace Tungsten::Gpu {
namespace {

constexpr unsigned int threadsPerBlock = 128;
constexpr float pi = 3.14159265358979323846f;

__device__ Float3 add(Float3 a, Float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ Float3 subtract(Float3 a, Float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ Float3 multiply(Float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ float dot(Float3 a, Float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ Float3 position(const RaySegmentInput &ray, float t) {
    return add(ray.origin, multiply(ray.direction, t));
}

// Same xxhash32(Vec2u) ordering and PCG sequence as Tungsten's CPU sampler.
__device__ unsigned int hash2(unsigned int x, unsigned int y)
{
    unsigned int h = y + 374761393U + x * 3266489917U;
    h = 668265263U * ((h << 17) | (h >> 15));
    h = 2246822519U * (h ^ (h >> 15));
    h = 3266489917U * (h ^ (h >> 13));
    return h ^ (h >> 16);
}

struct PcgSampler {
    unsigned long long state;

    __device__ unsigned int nextInt()
    {
        const unsigned long long old = state;
        state = old * 6364136223846793005ULL + 1ULL;
        const unsigned int shifted = static_cast<unsigned int>(((old >> 18U) ^ old) >> 27U);
        const unsigned int rotation = static_cast<unsigned int>(old >> 59U);
        return (shifted >> rotation) | (shifted << ((-rotation) & 31U));
    }

    __device__ float nextFloat()
    {
        return __uint_as_float((nextInt() >> 9U) | 0x3F800000U) - 1.0f;
    }

    __device__ explicit PcgSampler(unsigned int seed) : state(seed)
    {
        nextFloat();
        nextFloat();
    }
};

__device__ Float3 transformPoint(const Affine3x4 &matrix, Float3 p)
{
    return {
        matrix.m[0] * p.x + matrix.m[1] * p.y + matrix.m[2] * p.z + matrix.m[3],
        matrix.m[4] * p.x + matrix.m[5] * p.y + matrix.m[6] * p.z + matrix.m[7],
        matrix.m[8] * p.x + matrix.m[9] * p.y + matrix.m[10] * p.z + matrix.m[11]
    };
}

__device__ float voxel(const DeviceMeanGrid &grid, int x, int y, int z)
{
    x = max(0, min(x - grid.origin[0], grid.dimensions[0] - 1));
    y = max(0, min(y - grid.origin[1], grid.dimensions[1] - 1));
    z = max(0, min(z - grid.origin[2], grid.dimensions[2] - 1));
    const std::size_t index = (static_cast<std::size_t>(z) * grid.dimensions[1] + y) * grid.dimensions[0] + x;
    return grid.voxels[index];
}

__device__ float meanValue(const DeviceMeanGrid &grid, Float3 p)
{
    Float3 q = transformPoint(grid.worldToIndex, p);
    q.x = fminf(fmaxf(q.x, grid.clampMin.x), grid.clampMax.x);
    q.y = fminf(fmaxf(q.y, grid.clampMin.y), grid.clampMax.y);
    q.z = fminf(fmaxf(q.z, grid.clampMin.z), grid.clampMax.z);
    const int x = static_cast<int>(floorf(q.x));
    const int y = static_cast<int>(floorf(q.y));
    const int z = static_cast<int>(floorf(q.z));
    const float fx = q.x - x, fy = q.y - y, fz = q.z - z;

    float layer[2];
    for (int dz = 0; dz != 2; ++dz) {
        float row[2];
        for (int dy = 0; dy != 2; ++dy) {
            const float a = voxel(grid, x, y + dy, z + dz);
            const float b = voxel(grid, x + 1, y + dy, z + dz);
            row[dy] = a + fx * (b - a);
        }
        layer[dz] = row[0] + fy * (row[1] - row[0]);
    }
    return (layer[0] + fz * (layer[1] - layer[0]) + grid.offset) * grid.scale;
}

__device__ Float3 meanGradient(const DeviceMeanGrid &grid, Float3 p)
{
    constexpr float eps = 0.001f; // TabulatedMean::dmean_da() uses a forward difference.
    const float center = meanValue(grid, p);
    return {
        (meanValue(grid, add(p, {eps, 0, 0})) - center) / eps,
        (meanValue(grid, add(p, {0, eps, 0})) - center) / eps,
        (meanValue(grid, add(p, {0, 0, eps})) - center) / eps
    };
}

struct NoiseSample { float value, derivative; };

__device__ NoiseSample noiseAt(float coordinate, unsigned int seed,
                               const Se1DRayParameters &params, float normalization)
{
    const float cellCoordinate = coordinate / params.kernelRadius;
    const float cellFloor = floorf(cellCoordinate);
    const int cell = static_cast<int>(cellFloor);
    const float fraction = cellCoordinate - cellFloor;
    NoiseSample sum{0.0f, 0.0f};
    for (int offset = -1; offset <= 1; ++offset) {
        PcgSampler sampler(hash2(static_cast<unsigned int>(cell + offset), seed) + 1U);
        for (unsigned int k = 0; k < static_cast<unsigned int>(params.impulseDensity); ++k) {
            const float impulse = sampler.nextFloat();
            const float weight = sampler.nextFloat() < 0.5f ? -1.0f : 1.0f;
            const float delta = fraction - offset - impulse;
            if (delta * delta < 1.0f) {
                const SeSplat1DResult splat = evaluateSeSplat1D(params.kernelRadius * delta, 0.0f, 1.0f);
                sum.value += weight * splat.value;
                sum.derivative += weight * splat.derivative;
            }
        }
    }
    sum.value /= normalization;
    sum.derivative /= normalization;
    return sum;
}

struct Condition { float valueScale, gradientScale, originCoordinate; };

__device__ NoiseSample conditionedNoise(float coordinate, unsigned int seed,
                                        const Se1DRayParameters &params, float normalization,
                                        const Condition &condition)
{
    NoiseSample result = noiseAt(coordinate, seed, params, normalization);
    const float delta = coordinate - condition.originCoordinate;
    const float covariance = expf(-0.25f * delta * delta);
    const float first = -0.5f * delta * covariance;
    const float second = (0.25f * delta * delta - 0.5f) * covariance;
    result.value += condition.valueScale * covariance + condition.gradientScale * first;
    result.derivative += condition.valueScale * first + condition.gradientScale * second;
    return result;
}

__device__ float fieldValue(const DeviceMeanGrid &grid, const RaySegmentInput &ray, float t,
                            const Se1DRayParameters &params, float noiseScale,
                            float normalization, const Condition &condition)
{
    const Float3 p = position(ray, t);
    const float coordinate = dot(p, ray.direction) / noiseScale;
    return meanValue(grid, p) + params.amplitude *
        conditionedNoise(coordinate, ray.noiseSeed, params, normalization, condition).value;
}

__device__ Float3 hitGradient(const DeviceMeanGrid &grid, const RaySegmentInput &ray, float t,
                              const Se1DRayParameters &params, float noiseScale,
                              float normalization, const Condition &condition)
{
    const Float3 p = position(ray, t);
    const float coordinate = dot(p, ray.direction) / noiseScale;
    const float longitudinal = conditionedNoise(coordinate, ray.noiseSeed, params,
                                                normalization, condition).derivative;

    PcgSampler sampler(ray.normalSeed);
    // CPU evaluateGradientNoise1D draws xy_1, then independent xy_2 at a hit.
    const double firstU = sampler.nextFloat(), firstV = sampler.nextFloat();
    const double secondU = sampler.nextFloat(), secondV = sampler.nextFloat();
    const bool atOrigin = t == 0.0f;
    const double u = atOrigin ? firstU : secondU;
    const double v = atOrigin ? firstV : secondV;
    const double radius = sqrt(-2.0 * log(1.0 - u));
    const float x = static_cast<float>(radius * cos(2.0 * pi * v) * 0.7071067811865475);
    const float y = static_cast<float>(radius * sin(2.0 * pi * v) * 0.7071067811865475);

    const Float3 n = ray.direction;
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    const Float3 tangent{1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
    const Float3 bitangent{b, sign + n.y * n.y * a, -n.y};
    const Float3 local = add(add(multiply(tangent, x), multiply(bitangent, y)),
                             multiply(n, longitudinal));
    return add(meanGradient(grid, p), multiply(local, params.amplitude / noiseScale));
}

__device__ RaySegmentOutput traceOne(const RaySegmentInput &ray, const DeviceMeanGrid &grid,
                                     const Se1DRayParameters &params)
{
    RaySegmentOutput out{};
    out.status = RaySegmentStatus::InvalidInput;
    if (!isfinite(ray.nearT) || !isfinite(ray.farT) || ray.farT <= ray.nearT ||
        !isfinite(ray.jitter) || ray.jitter < 0.0f || ray.jitter >= 1.0f ||
        !isfinite(ray.origin.x) || !isfinite(ray.origin.y) || !isfinite(ray.origin.z) ||
        !isfinite(ray.direction.x) || !isfinite(ray.direction.y) || !isfinite(ray.direction.z) ||
        (!ray.firstScatter && (!isfinite(ray.previousValue) ||
            !isfinite(ray.previousGradient.x) || !isfinite(ray.previousGradient.y) ||
            !isfinite(ray.previousGradient.z))) ||
        fabsf(dot(ray.direction, ray.direction) - 1.0f) > 0.001f)
        return out;

    const float noiseScale = params.lengthScale * 0.7071067811865475f;
    const float normalization = sqrtf(params.impulseDensity / params.kernelRadius * sqrtf(pi));
    const float stepSize = fminf(params.maxStepSize, (ray.farT - ray.nearT) / params.minSteps);
    if (!(stepSize > 0.0f))
        return out;

    Condition condition{0.0f, 0.0f, dot(ray.origin, ray.direction) / noiseScale};
    if (!ray.firstScatter && params.amplitude != 0.0f) {
        const NoiseSample initial = noiseAt(condition.originCoordinate, ray.noiseSeed,
                                            params, normalization);
        const float initialMean = meanValue(grid, ray.origin);
        const Float3 initialMeanGrad = meanGradient(grid, ray.origin);
        condition.valueScale = (ray.previousValue - initialMean) / params.amplitude - initial.value;
        condition.gradientScale = -2.0f * (noiseScale *
            dot(subtract(ray.previousGradient, initialMeanGrad), ray.direction) /
            params.amplitude - initial.derivative);
    }

    const float initialValue = fieldValue(grid, ray, ray.nearT, params, noiseScale,
                                          normalization, condition);
    if (!isfinite(initialValue))
        return out;
    int initialSign = initialValue < 0.0f ? -1 : 1;
    float previousValue = initialValue;
    float t = ray.nearT + stepSize * ray.jitter;
    unsigned int steps = 0;
    while (t < ray.farT && steps < params.maxSteps) {
        ++steps;
        const float currentValue = fieldValue(grid, ray, t, params, noiseScale,
                                              normalization, condition);
        if (!isfinite(currentValue))
            return out;
        const int sign = currentValue < 0.0f ? -1 : 1;
        if (!ray.firstScatter && steps == 1) {
            initialSign = sign;
        } else if (sign != initialSign) {
            float factor = previousValue / (previousValue - currentValue);
            float candidate = t - stepSize + stepSize * factor;
            float previousCandidate = candidate;
            for (;;) {
                candidate = t - stepSize + stepSize * factor;
                const float value = fieldValue(grid, ray, candidate, params, noiseScale,
                                               normalization, condition);
                if (!isfinite(value))
                    return out;
                if ((value < 0.0f ? -1 : 1) == initialSign)
                    break;
                factor *= 0.9f;
                if (factor <= 0.01f) {
                    previousCandidate = 0.0f;
                    break;
                }
                previousCandidate = candidate;
            }
            out.t = previousCandidate;
            out.value = 0.0f;
            out.gradient = hitGradient(grid, ray, out.t, params, noiseScale,
                                       normalization, condition);
            if (!isfinite(out.gradient.x) || !isfinite(out.gradient.y) ||
                !isfinite(out.gradient.z)) {
                out.hit = 0;
                return out;
            }
            out.hit = 1;
            out.steps = steps;
            out.status = RaySegmentStatus::Ok;
            return out;
        }
        previousValue = currentValue;
        t += stepSize;
    }

    out.t = ray.farT;
    out.steps = steps;
    if (t < ray.farT) {
        out.status = RaySegmentStatus::StepLimit;
        return out;
    }
    out.value = fieldValue(grid, ray, ray.farT, params, noiseScale, normalization, condition);
    if (!isfinite(out.value))
        out.status = RaySegmentStatus::InvalidInput;
    else
        out.status = RaySegmentStatus::Ok;
    return out;
}

__global__ void traceSegments(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                              std::size_t count, DeviceMeanGrid grid,
                              Se1DRayParameters parameters)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        outputs[index] = traceOne(inputs[index], grid, parameters);
}

} // namespace

cudaError_t launchSe1DRaySegments(const RaySegmentInput *deviceInputs,
                                  RaySegmentOutput *deviceOutputs,
                                  std::size_t count,
                                  const DeviceMeanGrid &deviceGrid,
                                  const Se1DRayParameters &parameters,
                                  cudaStream_t stream)
{
    if (!deviceGrid.voxels || deviceGrid.dimensions[0] < 2 ||
        deviceGrid.dimensions[1] < 2 || deviceGrid.dimensions[2] < 2 ||
        !std::isfinite(parameters.lengthScale) || parameters.lengthScale <= 0.0f ||
        !std::isfinite(parameters.amplitude) ||
        !std::isfinite(parameters.kernelRadius) || parameters.kernelRadius <= 0.0f ||
        !std::isfinite(parameters.impulseDensity) || parameters.impulseDensity <= 0.0f ||
        parameters.impulseDensity > 64.0f ||
        !std::isfinite(parameters.maxStepSize) || parameters.maxStepSize <= 0.0f ||
        parameters.minSteps == 0 || parameters.maxSteps == 0)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    if (!deviceInputs || !deviceOutputs ||
        count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) * threadsPerBlock)
        return cudaErrorInvalidValue;
    const unsigned int blocks = static_cast<unsigned int>((count - 1) / threadsPerBlock + 1);
    traceSegments<<<blocks, threadsPerBlock, 0, stream>>>(deviceInputs, deviceOutputs,
                                                           count, deviceGrid, parameters);
    return cudaGetLastError();
}

} // namespace Tungsten::Gpu
