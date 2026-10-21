#include "SdfFunctions.hpp"
#include <math/Vec.hpp>
#include <math/Mat4f.hpp>
#include <math/MathUtil.hpp>
#include <math/Angle.hpp>
#include <Debug.hpp>

namespace Tungsten {

    std::string SdfFunctions::functionToString(SdfFunctions::Function val)
    {
        switch (val) {
        default:
        case Function::Knob:  return "knob";
        case Function::KnobInner:  return "knob_inner";
        case Function::KnobOuter:  return "knob_outer";
        case Function::TwoSpheres:  return "two_spheres";
        case Function::Plane:  return "plane";
        }
    }

    SdfFunctions::Function SdfFunctions::stringToFunction(const std::string& name)
    {
        if (name == "knob")
            return Function::Knob;
        else if (name == "knob_inner")
            return Function::KnobInner;
        else if (name == "knob_outer")
            return Function::KnobOuter;
        else if (name == "two_spheres")
            return Function::TwoSpheres;
        else if (name == "plane")
            return Function::Plane;
        FAIL("Invalid sdf function: '%s'", name);
    }

    /*
    Copyright 2020 Towaki Takikawa @yongyuanxi
    The MIT License
    Link: N/A
    */

    /******************************************************************************
     * The MIT License (MIT)
     * Copyright (c) 2021, NVIDIA CORPORATION.
     * Permission is hereby granted, free of charge, to any person obtaining a copy of
     * this software and associated documentation files (the "Software"), to deal in
     * the Software without restriction, including without limitation the rights to
     * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
     * the Software, and to permit persons to whom the Software is furnished to do so,
     * subject to the following conditions:
     * The above copyright notice and this permission notice shall be included in all
     * copies or substantial portions of the Software.
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
     * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
     * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
     * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
     * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
     ******************************************************************************/

    // Knob geometry and differential queries share one definition in SdfFunctionsQueries.cpp.

    float SdfFunctions::two_spheres(Vec3f p, int& mat) {
        return min((p - Vec3f(0., 10., 0.f)).length() - 9.5f, (p - Vec3f(0.f, -10.f, 0.f)).length() - 9.5f);
    }

    template<typename T>
    T fract(T v) {
        return v - floor(v);
    }

    template<typename Vec>
    auto dot(Vec a, Vec b) {
        return a.dot(b);
    }

    /* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
    Vec3f random3(Vec3f c) {
        float j = 4096.0 * sin(dot(c, Vec3f(17.0f, 59.4f, 15.0f)));
        Vec3f r;
        r.z() = fract(512.0 * j);
        j *= .125;
        r.x() = fract(512.0 * j);
        j *= .125;
        r.y() = fract(512.0 * j);
        return r - 0.5;
    }

    /* skew constants for 3d simplex functions */
    const float F3 = 0.3333333;
    const float G3 = 0.1666667;


    template<typename ElementType, unsigned Size>
    Tungsten::Vec<ElementType, Size> step(const Tungsten::Vec<ElementType, Size>& edge, const Tungsten::Vec<ElementType, Size>& x)
    {
        Tungsten::Vec<ElementType, Size> result;
        for (unsigned i = 0; i < Size; ++i)
            result[i] = x[i] < edge[i] ? 0 : 1;
        return result;
    }


    /* 3d simplex noise */
    float simplex3d(Vec3f p) {
        /* 1. find current tetrahedron T and it's four vertices */
        /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
        /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/

        /* calculate s and x */
        Vec3f s = std::floor(p + dot(p, Vec3f(F3)));
        Vec3f x = p - s + dot(s, Vec3f(G3));

        /* calculate i1 and i2 */
        Vec3f e = step(Vec3f(0.0), x - Vec3f(x.y(), x.z(), x.x()));
        Vec3f i1 = e * (1.0f - Vec3f(e.z(), e.x(), e.y()));
        Vec3f i2 = 1.0f - Vec3f(e.z(), e.x(), e.y()) * (1.0f - e);

        /* x1, x2, x3 */
        Vec3f x1 = x - i1 + G3;
        Vec3f x2 = x - i2 + 2.0f * G3;
        Vec3f x3 = x - 1.0f + 3.0f * G3;

        /* 2. find four surflets and store them in d */
        Vec4f w, d;

        /* calculate surflet weights */
        w.x() = dot(x, x);
        w.y() = dot(x1, x1);
        w.z() = dot(x2, x2);
        w.w() = dot(x3, x3);

        /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
        w = max(0.6f - w, Vec4f(0.0f));

        /* calculate surflet components */
        d.x() = dot(random3(s), x);
        d.y() = dot(random3(s + i1), x1);
        d.z() = dot(random3(s + i2), x2);
        d.w() = dot(random3(s + 1.0), x3);

        /* multiply d by w^4 */
        w *= w;
        w *= w;
        d *= w;

        /* 3. return the sum of the four surflets */
        return dot(d, Vec4f(52.0f));
    }

    // fbm function by https://code.google.com/p/fractalterraingeneration/wiki/Fractional_Brownian_Motion
    double fbm(Vec3d uv, int octaves)
    {
        float gain = 0.65;
        float lacunarity = 2.1042;

        float total = 0.0;
        float frequency = 0.5;
        float amplitude = gain;

        uv = uv * 5.0;

        total = simplex3d(Vec3f(uv));

        for (int i = 0; i < octaves; i++)
        {
            total += simplex3d(Vec3f(uv) * frequency) * amplitude;
            frequency *= lacunarity;
            amplitude *= gain;
        }

        total = (total + 2.0) / 4.0;

        return total;
    }


}