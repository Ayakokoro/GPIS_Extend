// Lattice cubic GPIS | Shadertoy: paste into Image, no channels required.
// Left: trilinear mean. Right: one coherent family of 1-D GP rays.
// Click the shell or pedestal in either panel to inspect that ray in the plot.
// For the progressive ensemble, prepend #define ENSEMBLE_BUFFER and use in Buffer A.
precision highp float;
precision highp int;

#define NONSTATIONARY 0
#define THREE_SIGMA 0
#define CONDITION_STATIONARY_BIRTH 0
#define SMOOTH_MEAN_NORMALS 1

const uint REALIZATION = 23u;
const float SIGMA = 0.082;
const float SPACING = 0.11;       // delta = ell/2 (or ell_min/2).
// Native 609^3 shaderball_sdf.vdb, represented in coordinates relative to the
// original knob center (world 0,0.8,0). Four halo voxels enclose the process box.
const float VOXEL = 0.005;
const vec3 GRID_ORIGIN = vec3(-1.52,-1.31899996519,-1.52);
const vec3 BOX_MIN = GRID_ORIGIN + vec3(4.0*VOXEL);
const vec3 BOX_MAX = GRID_ORIGIN + vec3(604.0*VOXEL);
const int MAX_VOXEL_SPANS = int(1.733*SPACING/VOXEL) + 6;
// Half this flight length encloses the farthest process-box corner from the
// knob center at every camera angle. Changing the kernel spacing keeps the box.
const float FAR_T = 5.72;
const int NOISE_CELLS = int(ceil(FAR_T/SPACING));

// Rotate each Gaussian pair by the same deterministic, ray-dependent angle.
// This preserves the per-ray Gaussian law and couples nearby parallel rays.
vec2 rayRotation;
vec2 birthCorrection;
uint flightSeed;


// Stationary SE ray noise. Lattice spacing = ell / 2 and kernel radius = 3 ell.
// The weights correspond to unit sigma. Derivatives are per lattice cell.
const float LATTICE_WEIGHT[7] = float[7](
    0.631618778, 0.491905199, 0.232359563, 0.066572129,
    0.011568501, 0.001219311, 0.0
);

vec2 latticeWeight(int offset) {
    float b = LATTICE_WEIGHT[min(abs(offset), 6)];
    return vec2(b, -0.5 * float(offset) * b);
}

uint latticeHash(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

// Two neighboring source indices share the same Box-Muller evaluation.
// Open uniforms exclude 0 and 1 even on a 24-bit float mantissa.
vec2 latticeGaussianPair(int pairIndex, uint seed) {
    uint key = (2u * uint(pairIndex)) ^ seed;
    vec2 u = (vec2(latticeHash(key) >> 9u, latticeHash(key ^ 1u) >> 9u)
              + 0.5) * (1.0 / 8388608.0);
    float angle = 6.28318530718 * u.y;
    vec2 pair = sqrt(-2.0 * log(u.x)) * vec2(cos(angle), sin(angle));
    return vec2(dot(rayRotation, vec2(pair.x, -pair.y)),
                dot(rayRotation, pair.yx));
}

// correction = (0,0) for the prior. Only sources overlapping knot 0 change.
vec2 latticeSources(int pairIndex, uint seed, vec2 correction) {
    int j = pairIndex * 2;
    vec2 z = latticeGaussianPair(pairIndex, seed);
    if (j >= -6 && j <= 4) {
        z += vec2(dot(latticeWeight(-j), correction),
                  dot(latticeWeight(-j - 1), correction));
    }
    return z;
}

// Produces noise value and derivative with respect to cell coordinates.
vec2 latticeKnot(int knot, uint seed, float sigma, vec2 correction) {
    int firstPair = (knot - 5) >> 1;
    vec2 jet = vec2(0.0);
    for (int k = 0; k < 6; ++k) {
        int pairIndex = firstPair + k;
        int offset = knot - 2 * pairIndex;
        vec2 z = latticeSources(pairIndex, seed, correction);
        jet += latticeWeight(offset) * z.x + latticeWeight(offset - 1) * z.y;
    }
    return sigma * jet;
}

// target = desired noise value and physical directional derivative at t = 0,
// after subtracting the mean field and its derivative. Supply an already
// scrambled flight seed (latticeHash(flightID) is one choice).
vec2 latticeBirthUpdate(uint seed, float sigma, float spacing, vec2 target) {
    if (sigma <= 0.0) return vec2(0.0);
    vec2 prior = latticeKnot(0, seed, 1.0, vec2(0.0));
    // The even value row and odd slope row are orthogonal at the birth knot.
    return (vec2(target.x, target.y * spacing) / sigma - prior)
           * vec2(1.000000007, 4.000002577);
}

vec4 latticeHermite(vec2 leftJet, vec2 rightJet) {
    float dy = rightJet.x - leftJet.x;
    return vec4(leftJet, 3.0 * dy - 2.0 * leftJet.y - rightJet.y,
                leftJet.y + rightJet.y - 2.0 * dy);
}

// Assemble an arbitrary noise segment in one pass. In a consecutive scan,
// reuse the previous rightJet as leftJet and call latticeKnot only once per cell.
vec4 latticePiece(int cell, uint seed, float sigma, vec2 correction) {
    int firstPair = (cell - 5) >> 1;
    vec4 jets = vec4(0.0);
    for (int k = 0; k < 7; ++k) {
        int pairIndex = firstPair + k;
        int offset = cell - 2 * pairIndex;
        vec2 z = latticeSources(pairIndex, seed, correction);
        vec2 a = latticeWeight(offset - 1);
        vec2 b = latticeWeight(offset);
        vec2 c = latticeWeight(offset + 1);
        jets += z.x * vec4(b, c) + z.y * vec4(a, b);
    }
    return latticeHermite(sigma * jets.xy, sigma * jets.zw);
}

#if NONSTATIONARY
// A one-dimensional nonstationary SE process convolution. Distances in this
// function are measured in knot intervals; multiply its derivatives by
// 1/spacing for the physical ray derivative. sigma is a scene amplitude cap.
vec4 varyingPiece(int cell, uint seed, float spacing, float sigma) {
    // A C1 parameter ramp from the third to the sixteenth knot. Physical
    // ell goes from 2*spacing to 3.4*spacing; sigma goes from .65*sigma to sigma.
    vec2 t = (vec2(float(cell), float(cell + 1)) * spacing - 3.0 * spacing)
             / (13.0 * spacing);
    t = clamp(t, vec2(0.0), vec2(1.0));
    vec2 ramp = t * t * (3.0 - 2.0 * t);
    vec2 rampSlope = (6.0 / 13.0) * t * (1.0 - t);
    vec2 width = 2.0 + 1.4 * ramp; // ell / spacing
    vec2 widthSlope = 1.4 * rampSlope;
    vec2 amp = sigma * (0.65 + 0.35 * ramp);
    vec2 ampSlope = (sigma * 0.35) * rampSlope;

    // Both endpoints share each contributing Gaussian coefficient. A source at
    // distance exactly 3 ell has zero value and derivative. At most 12 pairs.
    vec2 invWidth = 1.0 / width;
    vec2 unit = 0.893243842 * inversesqrt(width); // (2/pi)^(1/4) sqrt(spacing/ell)
    vec2 ampJet = ampSlope - 0.5 * amp * widthSlope * invWidth;
    int reach = int(floor(3.0 * max(width.x, width.y)));
    int firstPair = (cell - reach) >> 1;
    int lastPair = (cell + 1 + reach) >> 1;

    // Consecutive exp(-x*x) factors have geometric successive ratios.
    // Six exponentials initialize both endpoint walks for the whole segment.
    vec2 offsets = vec2(float(cell - 2 * firstPair), float(cell + 1 - 2 * firstPair));
    vec2 inverseSquare = invWidth * invWidth;
    vec2 gaussian = exp(-offsets * offsets * inverseSquare);
    vec2 ratio = exp((2.0 * offsets - 1.0) * inverseSquare);
    vec2 ratioDecay = exp(-2.0 * inverseSquare);
    vec2 values = vec2(0.0), slopes = vec2(0.0);
    for (int n = 0; n < 12; ++n) {
        int pairIndex = firstPair + n;
        if (pairIndex > lastPair) break;
        vec2 pairZ = latticeGaussianPair(pairIndex, seed);
        for (int lane = 0; lane < 2; ++lane) {
            vec2 x = offsets * invWidth;
            // The final 0.5 ell of the 3 ell support smoothly goes to zero.
            vec2 v = clamp(2.0 * (3.0 - abs(x)), vec2(0.0), vec2(1.0));
            vec2 window = v * v * (3.0 - 2.0 * v);
            vec2 windowDx = -12.0 * sign(x) * v * (1.0 - v);
            vec2 dx = (1.0 - x * widthSlope) * invWidth;
            vec2 weighted = pairZ[lane] * gaussian * unit;
            values += weighted * amp * window;
            slopes += weighted * (ampJet * window + amp * dx * (windowDx - 2.0 * x * window));
            gaussian *= ratio;
            ratio *= ratioDecay;
            offsets -= 1.0;
        }
    }
    return latticeHermite(vec2(values.x, slopes.x), vec2(values.y, slopes.y));
}
#endif

// Power coefficients in ascending order, on a normalized segment [0,1).
float downcrossingValue(vec4 c, float u)
{
    return ((c.w * u + c.z) * u + c.y) * u + c.x;
}

float interiorCut(float x)
{
    return x > 0.0 && x < 1.0 ? x : 1.0;
}

bool firstDowncrossing(vec4 c, inout bool positiveOnLeft, out float u)
{
    u = 0.0;
    float endValue = downcrossingValue(c, 1.0);
    vec4 control = vec4(c.x, c.x + c.y / 3.0,
                        c.x + (2.0 * c.y + c.z) / 3.0, endValue);
    float margin = 8e-7 * dot(abs(c), vec4(1.0));
    if (all(greaterThan(control, vec4(margin)))) {
        positiveOnLeft = true;
        return false;
    }
    if (all(lessThan(control, vec4(-margin)))) {
        positiveOnLeft = false;
        return false;
    }

    // Split at stationary points. The q form avoids cancellation when the
    // two roots of the derivative have very different magnitudes.
    vec2 cuts = vec2(1.0);
    vec3 deriv = vec3(3.0 * c.w, 2.0 * c.z, c.y);
    float scale = max(abs(deriv.x), max(abs(deriv.y), abs(deriv.z)));
    if (scale > 1e15 || (scale > 0.0 && scale < 1e-15)) deriv /= scale;
    if (deriv.x == 0.0) {
        if (deriv.y != 0.0) cuts.x = interiorCut(-deriv.z / deriv.y);
    } else {
        float disc = deriv.y * deriv.y - 4.0 * deriv.x * deriv.z;
        if (disc > 0.0) {
            float sd = sqrt(disc);
            float q = -0.5 * (deriv.y + (deriv.y < 0.0 ? -sd : sd));
            float r0 = interiorCut(q / deriv.x);
            float r1 = interiorCut(deriv.z / q);
            cuts = vec2(min(r0, r1), max(r0, r1));
        } else if (disc == 0.0) {
            cuts.x = interiorCut(-0.5 * deriv.y / deriv.x);
        }
    }

    float lo = 0.0;
    float left = c.x;
    for (int part = 0; part < 3; ++part) {
        float hi = part == 0 ? cuts.x : (part == 1 ? cuts.y : 1.0);
        float right = hi == 1.0 ? endValue : downcrossingValue(c, hi);
        if (right < 0.0 && (left > 0.0 || (left == 0.0 && positiveOnLeft))) {
            if (left == 0.0) {
                u = lo;
                return true;
            }
            // Only a hit pays for refinement; interval width at most 2^-16.
            for (int step = 0; step < 16; ++step) {
                float mid = 0.5 * (lo + hi);
                float fm = downcrossingValue(c, mid);
                if (fm == 0.0) { u = mid; return true; }
                if (fm > 0.0) lo = mid;
                else hi = mid;
            }
            u = 0.5 * (lo + hi);
            return true;
        }
        if (right != 0.0) positiveOnLeft = right > 0.0;
        else if (left != 0.0) positiveOnLeft = left > 0.0;
        if (hi == 1.0) break;  // Shared endpoints are decided by the next piece.
        lo = hi;
        left = right;
    }
    return false;
}

vec4 rayNoise(int cell) {
#if NONSTATIONARY
    return varyingPiece(cell, flightSeed, SPACING, SIGMA);
#else
    return latticePiece(cell, flightSeed, SIGMA, birthCorrection);
#endif
}

void updateBirth() {
    birthCorrection = vec2(0.0);
#if CONDITION_STATIONARY_BIRTH && !NONSTATIONARY
    // Zero noise value and physical slope at birth (the total equals the mean).
    birthCorrection = latticeBirthUpdate(flightSeed, SIGMA, SPACING, vec2(0.0));
#endif
}

float polySlope(vec4 c, float u) { return (3.0*c.w*u + 2.0*c.z)*u + c.y; }

// Geometry from Tungsten SdfFunctionsQueries.cpp, knobGeometry; display origin
// is native world (0, 0.8, 0). Original knob SDF: Towaki Takikawa (2020).
/*
The MIT License (MIT)
Copyright (c) 2021, NVIDIA CORPORATION.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
float knobUnion(float a,float b,float k) {
    float h=clamp(0.5+0.5*(b-a)/k,0.0,1.0);
    return mix(b,a,h)-k*h*(1.0-h);
}
float knobSubtract(float cut,float shape,float k) {
    return -knobUnion(cut,-shape,k);
}
float knobCylinder(vec3 p,vec2 extent) {
    vec2 d=vec2(length(p.xz),abs(p.y))-extent;
    return min(max(d.x,d.y),0.0)+length(max(d,vec2(0)));
}
vec3 knobRotateX(vec3 p,float c,float s) {
    return vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);
}
float meanNode(vec3 p) {
    p*=1.25;
    float outer=length(p)-1.0;
    float cutout=length(p-vec3(0,0.5,0.5))-0.7;
    const float cos45=0.70710677, cos90=-4.37113883e-8;
    vec3 torus=knobRotateX(p-vec3(0,0.2,0.2),cos45,-cos45);
    float etch=length(vec2(length(torus.xz)-1.0,torus.y))-0.05;
    float body=min(knobSubtract(cutout,outer,0.1),length(p)-0.75);
    body=max(body,-etch);

    vec3 b=p+vec3(0,0.775,0);
    vec3 bottomCone=knobRotateX(b+vec3(0,0.9,0),cos90,-1.0);
    vec3 topCone=knobRotateX(b-vec3(0,0.9,0),cos90,1.0);
    const float coneSlope=1.04719758;
    float base=knobUnion(coneSlope*(length(bottomCone.xy)+bottomCone.z),
                         coneSlope*(length(topCone.xy)+topCone.z),0.02);
    base=0.7*max(base,knobCylinder(b,vec2(1.1,0.25)));
    base=max(base,-knobCylinder(b,vec2(0.6,0.3)));
    vec3 q=knobRotateX(b-vec3(0,0,1.0),cos90,1.0);
    float notch=max(abs(q.z)-0.3,max(abs(q.x)*0.866025+q.y*0.5,-q.y)-0.6);
    base=max(base,-notch);
    return 0.8*min(knobSubtract(outer,base,0.1),body);
}

// Studio shading only: the continuous node generator supplies a smooth mean
// normal. The hit distance and the diagnostic curves always use the trilinear field.
vec3 smoothMeanGradient(vec3 p) {
    const float e=VOXEL;
    return vec3(meanNode(p+vec3(e,0,0))-meanNode(p-vec3(e,0,0)),
                meanNode(p+vec3(0,e,0))-meanNode(p-vec3(0,e,0)),
                meanNode(p+vec3(0,0,e))-meanNode(p-vec3(0,0,e)))/(2.0*e);
}

// x,y,z,w are the 00,10,01,11 corners of an xy face.
struct VoxelData { vec4 low; vec4 high; };
vec4 nodeFace(vec3 p) {
    return vec4(meanNode(p), meanNode(p+vec3(VOXEL,0,0)),
                meanNode(p+vec3(0,VOXEL,0)), meanNode(p+vec3(VOXEL,VOXEL,0)));
}
VoxelData loadVoxel(vec3 index) {
    vec3 p = GRID_ORIGIN+index*VOXEL;
    return VoxelData(nodeFace(p),nodeFace(p+vec3(0,0,VOXEL)));
}
// xyz = world-space gradient of the *trilinear* mean, w = value.
vec4 meanJet(VoxelData v, vec3 f) {
    f = clamp(f, 0.0, 1.0);
    vec4 s = mix(v.low,v.high,f.z);
    vec2 alongX = mix(s.xz,s.yw,f.x);
    vec4 dz = v.high-v.low;
    vec2 zSides = mix(dz.xz,dz.yw,f.x);
    vec3 gradient = vec3(mix(s.y-s.x,s.w-s.z,f.y),alongX.y-alongX.x,
                         mix(zSides.x,zSides.y,f.y))/VOXEL;
    return vec4(gradient,mix(alongX.x,alongX.y,f.y));
}
vec4 meanAt(vec3 p) {
    vec3 q = (p-GRID_ORIGIN)/VOXEL;
    return meanJet(loadVoxel(floor(q)), fract(q));
}

void camera(float viewTime,out vec3 dir,out vec3 right,out vec3 up) {
    float az = 0.28 + 0.42*sin(0.19*viewTime);
    dir = normalize(vec3(-sin(az),-0.245,-cos(az)));
    right = normalize(cross(dir,vec3(0,1,0)));
    up = cross(right,dir);
}
vec3 flightOrigin(vec2 plane,vec3 rd,vec3 ex,vec3 ey) {
    return plane.x*ex+plane.y*ey-rd*(0.5*FAR_T);
}
vec3 phaseJet(vec2 plane) {
    // theta(plane), dtheta/dx, dtheta/dy. Coordinates stay fixed on a ray.
    float v = dot(plane,vec2(2.6,-2.3));
    return vec3(2.4*plane.x+2.1*plane.y+0.85*sin(v),vec2(2.4,2.1)+0.85*cos(v)*vec2(2.6,-2.3));
}
void beginFlight(float angle,uint realization) {
    flightSeed = latticeHash(realization);
    rayRotation = vec2(cos(angle),sin(angle));
    updateBirth();
}

struct Contact { float t; vec3 grad; int cell; vec4 noise; };
vec2 processSpan(vec3 ro,vec3 rd) {
    vec2 span=vec2(0.0,FAR_T);
    for(int k=0;k<3;++k) {
        if(abs(rd[k])<1e-8) {
            if(ro[k]<BOX_MIN[k] || ro[k]>BOX_MAX[k]) return vec2(1,0);
        } else {
            vec2 faces=(vec2(BOX_MIN[k],BOX_MAX[k])-ro[k])/rd[k];
            span=vec2(max(span.x,min(faces.x,faces.y)),min(span.y,max(faces.x,faces.y)));
        }
    }
    return span;
}
Contact traceShaderball(vec3 ro,vec3 rd,vec2 span,bool noisy) {
    Contact miss = Contact(-1.0,vec3(0),0,vec4(0));
    if(span.x>=span.y) return miss;
    bool positiveOnLeft = true;
    vec3 inv = vec3(1.0)/mix(rd,vec3(1.0),lessThan(abs(rd),vec3(1e-8)));
    // The continuous knob is 1.04-Lipschitz. Interpolation changes its value
    // by at most L*sqrt(3)*VOXEL/2 (weighted RMS corner distance).
    const float L=1.04, interpPad=L*0.866026*VOXEL+0.00001;
    // Include the preceding index because the entry quotient can round at a knot.
    int firstCell=max(0,int(floor(span.x/SPACING))-1);
    for(int i=firstCell;i<NOISE_CELLS;++i) {
        float cellStart=float(i)*SPACING, b=min(float(i+1)*SPACING,span.y);
        float a=max(cellStart,span.x);
        if(cellStart>=span.y) break;
        if(a>=b) continue;
        float lower=meanNode(ro+(0.5*(a+b))*rd)-0.5*L*(b-a)-interpPad;
        if(!noisy && lower>0.0) { positiveOnLeft=true; continue; }
#if THREE_SIGMA
        if(noisy && a>span.x && lower>3.0*SIGMA) { positiveOnLeft=true; continue; }
#endif
        vec4 noise = vec4(0);
        if(noisy) noise = rayNoise(i);
        vec4 control = vec4(noise.x,noise.x+noise.y/3.0,
                            noise.x+(2.0*noise.y+noise.z)/3.0,downcrossingValue(noise,1.0));
        if(lower + min(min(control.x,control.y),min(control.z,control.w)) > 0.0) {
            positiveOnLeft=true;
            continue;
        }
        // Use the Lipschitz bound on the continuous generator plus ray noise
        // to skip certified positive prefixes before loading fine voxel corners.
        vec3 dc=abs(control.yzw-control.xyz);
        float totalSlope=L+3.0*max(dc.x,max(dc.y,dc.z))/SPACING;
        for(int j=0;j<5;++j) {
            float f=meanNode(ro+a*rd)+downcrossingValue(noise,(a-cellStart)/SPACING);
            float advance=(f-interpPad)/totalSlope;
            if(advance<VOXEL) break;
            a=min(b,a+advance);
            positiveOnLeft=true;
            if(a>=b) break;
        }
        if(a>=b) continue;
        vec3 gridPos=(ro+a*rd-GRID_ORIGIN)/VOXEL;
        vec3 index=floor(gridPos), direction=sign(rd);
        vec3 delta=abs(inv)*VOXEL;
        vec3 nextFace=(GRID_ORIGIN+VOXEL*(index+step(vec3(0),rd))-ro)*inv;
        nextFace=mix(nextFace,vec3(1e18),lessThan(abs(rd),vec3(1e-8)));
        float t=a;
        // A length-SPACING ray crosses at most 3+ceil(sqrt(3)*SPACING/VOXEL) faces.
        for(int j=0;j<MAX_VOXEL_SPANS;++j) {
            float end=min(b,min(nextFace.x,min(nextFace.y,nextFace.z)));
            if(end>t) {
                VoxelData v=loadVoxel(index);
                vec3 p=ro+t*rd, q=ro+end*rd;
                vec3 corner=GRID_ORIGIN+VOXEL*index;
                vec4 left=meanJet(v,(p-corner)/VOXEL);
                vec4 right=meanJet(v,(q-corner)/VOXEL);
                float u0=(t-cellStart)/SPACING, u1=(end-cellStart)/SPACING, width=end-t;
                vec2 l=vec2(left.w+downcrossingValue(noise,u0),
                            width*(dot(left.xyz,rd)+polySlope(noise,u0)/SPACING));
                vec2 r=vec2(right.w+downcrossingValue(noise,u1),
                            width*(dot(right.xyz,rd)+polySlope(noise,u1)/SPACING));
                // Both terms are cubics here; Hermite reconstruction of the sum is exact.
                float u=0.0;
                bool hit=(t==span.x && l.x<0.0);
                if(!hit) hit=firstDowncrossing(latticeHermite(l,r),positiveOnLeft,u);
                if(hit) {
                    float th=t+width*u;
                    vec3 gradient=meanJet(v,(ro+th*rd-corner)/VOXEL).xyz;
                    return Contact(th,gradient,i,noise);
                }
            }
            if(end>=b) break;
            bvec3 crossed=lessThanEqual(nextFace,vec3(end));
            index+=mix(vec3(0),direction,crossed);
            nextFace+=mix(vec3(0),delta,crossed);
            t=max(t,end);
        }
    }
    return miss;
}


// Distances here are screen pixels. Keep steep portions of the curves antialiased.
float flightPlotCurve(float y, float center, float value, float scale, float halfWidth)
{
    float delta = y - (center + value * scale);
    float normalScale = max(0.75, length(vec2(dFdx(delta), dFdy(delta))));
    return 1.0 - smoothstep(halfWidth - 0.45, halfWidth + 0.55, abs(delta) / normalScale);
}

vec3 plotFlight(vec2 xy, vec2 size, float worldT, float tMax,
                float meanVal, float totalVal, float tHit, float spacing)
{
    float mid = 0.5 * size.y;
    float scale = 0.46 * size.y; // At +/-1, leave a small margin.
    vec3 color = mix(vec3(0.046, 0.065, 0.088), vec3(0.067, 0.091, 0.115),
                     clamp(xy.y / max(size.y, 1.0), 0.0, 1.0));
    float pixelsPerT = size.x / max(tMax, 1e-5);
    float knotStride = max(spacing, 1e-5);
    float knotDx = abs(fract(worldT / knotStride + 0.5) - 0.5) * knotStride * pixelsPerT;
    float knot = (1.0 - smoothstep(0.0, 1.0, knotDx))
               * smoothstep(3.0, 10.0, knotStride * pixelsPerT);
    float tick = 1.0 - smoothstep(3.0, 6.0, abs(xy.y - mid));
    color = mix(color, vec3(0.41, 0.52, 0.60), knot * (0.09 + 0.28 * tick));
    float baseline = 1.0 - smoothstep(0.15, 1.10, abs(xy.y - mid));
    color = mix(color, vec3(0.44, 0.55, 0.61), 0.67 * baseline);

    float dash = 1.0 - smoothstep(2.7, 3.5, abs(mod(xy.x + 5.0, 10.0) - 5.0));
    float meanLine = flightPlotCurve(xy.y, mid, meanVal, scale, 0.9);
    float totalLine = flightPlotCurve(xy.y, mid, totalVal, scale, 1.2);
    color = mix(color, vec3(0.58, 0.73, 0.83), dash * meanLine);
    color = mix(color, vec3(1.00, 0.77, 0.39), totalLine);

    if (tHit >= 0.0 && tHit <= tMax) {
        vec2 hitDelta = vec2((worldT - tHit) * pixelsPerT, xy.y - mid);
        float guide = 1.0 - smoothstep(0.2, 1.15, abs(hitDelta.x));
        float dotMask = 1.0 - smoothstep(2.5, 3.6, length(hitDelta));
        color = mix(color, vec3(1.00, 0.47, 0.37), max(0.68 * guide, dotMask));
    }
    return color;
}

// Small in-image labels: MEAN / GPIS, five pixels high.
float panelLabel(vec2 p,bool noisy) {
    if(any(lessThan(p,vec2(0))) || p.x>=23.0 || p.y>=5.0) return 0.0;
    const uint glyphs[8]=uint[8](0x118d771u, 0x1f0bc3fu, 0x118fe2eu, 0x11cd671u, 0x1e8f43eu, 0x10be2fu, 0x1f2109fu, 0xf8383eu);
    ivec2 xy=ivec2(p);
    int bit=xy.x%6;
    if(bit==5) return 0.0;
    uint glyph=glyphs[(noisy?4:0)+xy.x/6];
    return float((glyph>>uint(5*xy.y+bit))&1u);
}


// A studio environment is visible on rays that miss the entire process volume.
vec3 backdrop(vec2 plane) {
    float fill=exp(-0.72*dot(plane-vec2(-0.2,0.25),plane-vec2(-0.2,0.25)));
    return mix(vec3(0.036,0.051,0.066),vec3(0.132,0.162,0.166),fill);
}
vec3 studio(vec3 n,vec3 rd) {
    vec3 light=normalize(vec3(-0.55,0.87,0.60));
    vec3 r=reflect(rd,n);
    float key=max(0.0,dot(n,light));
    float fill=max(0.0,dot(n,normalize(vec3(0.8,0.32,-0.65))));
    float sky=0.5+0.5*n.y;
    vec3 lacquer=vec3(0.018,0.245,0.178);
    vec3 color=lacquer*(0.21+0.38*sky+1.15*key)+vec3(0.14,0.20,0.22)*fill*0.65;
    float grazing=pow(1.0-clamp(dot(n,-rd),0.0,1.0),4.0);
    // Two fixed softboxes and a broad window reflection.
    float boxLight=pow(max(dot(r,normalize(vec3(-0.44,0.75,0.49))),0.0),34.0);
    float edgeLight=pow(max(dot(r,normalize(vec3(0.82,0.35,-0.48))),0.0),20.0);
    float window=pow(max(dot(r,normalize(vec3(-0.45,0.62,-0.64))),0.0),4.0);
    return color+(0.4+0.6*grazing)*(vec3(2.4,2.2,1.8)*boxLight
              +vec3(1.4,1.9,2.1)*edgeLight+vec3(0.58,0.65,0.65)*window)+0.08*grazing;
}
vec3 displayRGB(vec3 linearColor) {
    return pow(vec3(1.0)-exp(-1.5*max(linearColor,vec3(0))),vec3(1.0/2.2));
}

vec2 panelPoint(vec2 pixel,float strip,float panelW,float unit) {
    return vec2(mod(pixel.x,panelW)-0.5*panelW,
                pixel.y-strip-0.5*(iResolution.y-strip))*unit;
}

vec2 selectedProbe(float strip,float panelW,float unit) {
    if(abs(iMouse.z)>0.0 && iMouse.y>strip) return panelPoint(iMouse.xy,strip,panelW,unit);
    return vec2(0.34,0.13);
}

// Radiance of the first hit anywhere on the complete field (including the base).
vec3 renderFlight(vec2 plane,vec3 rd,vec3 ex,vec3 ey,uint realization,bool noisy) {
    vec3 phase=phaseJet(plane);
    beginFlight(phase.x,realization);
    vec3 ro=flightOrigin(plane,rd,ex,ey);
    Contact hit=traceShaderball(ro,rd,processSpan(ro,rd),noisy);
    if(hit.t<0.0) return backdrop(plane);
    vec3 n=hit.grad;
#if SMOOTH_MEAN_NORMALS
    n=smoothMeanGradient(ro+hit.t*rd);
#endif
    if(noisy) {
        float u=hit.t/SPACING-float(hit.cell);
        // Differentiate the Gaussian rotation across parallel rays for shading.
        rayRotation=vec2(-rayRotation.y,rayRotation.x); updateBirth();
        float transverse=downcrossingValue(rayNoise(hit.cell),u);
        n+=rd*(polySlope(hit.noise,u)/SPACING)+transverse*(phase.y*ex+phase.z*ey);
    }
    n=dot(n,n)>1e-12 ? normalize(n) : -rd;
    return studio(n,rd);
}

#ifndef ENSEMBLE_BUFFER
void mainImage(out vec4 fragColor,in vec2 fragCoord) {
    vec2 res=iResolution.xy;
    float strip=floor(0.24*res.y), panelW=0.5*res.x;
    float unit=2.39/min(res.y-strip,panelW);
    vec2 probe=selectedProbe(strip,panelW,unit);
    vec3 rd,ex,ey; camera(iTime,rd,ex,ey);
    vec3 rgb;
    if(fragCoord.y<strip) {
        vec3 ro=flightOrigin(probe,rd,ex,ey);
        beginFlight(phaseJet(probe).x,REALIZATION);
        vec2 span=processSpan(ro,rd);
        Contact hit=traceShaderball(ro,rd,span,true);
        float t=FAR_T*fragCoord.x/res.x;
        int cell=min(int(floor(t/SPACING)),NOISE_CELLS-1);
        bool inside=t>=span.x && t<=span.y;
        float mean=inside ? meanAt(ro+t*rd).w : 0.0;
        float y=downcrossingValue(rayNoise(cell),t/SPACING-float(cell));
        rgb=plotFlight(fragCoord,vec2(res.x,strip),t,FAR_T,mean,mean+y,hit.t,SPACING);
        if(!inside) rgb=vec3(0.055,0.075,0.091);
    } else {
        bool noisy=fragCoord.x>=panelW;
        vec2 plane=panelPoint(fragCoord,strip,panelW,unit);
        rgb=displayRGB(renderFlight(plane,rd,ex,ey,REALIZATION,noisy));
        // The ring in both views identifies the ray used in the plot.
        float distancePx=length(plane-probe)/unit;
        rgb=mix(rgb,vec3(1.0,0.76,0.46),0.83*(1.0-smoothstep(0.5,1.25,abs(distancePx-5.0))));
        float seam=abs(fragCoord.x-panelW);
        rgb=mix(rgb,vec3(0.16,0.21,0.24),1.0-smoothstep(0.1,1.0,seam));
        rgb=mix(rgb,vec3(0.31,0.37,0.39),1.0-smoothstep(0.1,1.3,fragCoord.y-strip));
        float labelScale=max(1.0,floor(res.y/230.0));
        vec2 local=vec2(mod(fragCoord.x,panelW)/labelScale-9.0,
                        (res.y-fragCoord.y)/labelScale-9.0);
        float label=panelLabel(local,noisy);
        rgb=mix(rgb,vec3(0.91,0.96,0.94),label);
    }
    fragColor=vec4(rgb,1.0);
}
#else

// Ensemble Buffer A. Connect iChannel0 to Buffer A (the previous frame).
// Accumulate for a fixed camera; alpha of each visible pixel is its sample count.
const int ENSEMBLE_TARGET = 128;       // 1..1024; integer-exact even in RGBA16F.
const float ENSEMBLE_VIEW_TIME = 0.0; // freeze the orbit at this time (seconds).

uint sampleRealization(int index) {
    // An odd Weyl increment visits distinct uint keys, followed by latticeHash in beginFlight.
    return REALIZATION+0x9e3779b9u*uint(index);
}

// Byte-sized lanes preserve the signature in both half and full float feedback.
vec4 feedbackKey(uint h) {
    return vec4(uvec4(h,h>>8u,h>>16u,h>>24u)&uvec4(255u));
}
vec4 sceneKey(ivec2 size) {
    uint h=latticeHash(uint(size.x))^uint(size.y);
    h=latticeHash(h)^REALIZATION;
    h=latticeHash(h)^floatBitsToUint(ENSEMBLE_VIEW_TIME);
    h=latticeHash(h)^floatBitsToUint(SIGMA);
    h=latticeHash(h)^floatBitsToUint(SPACING);
    h=latticeHash(h)^floatBitsToUint(VOXEL);
    h=latticeHash(h)^uint(NOISE_CELLS);
    h=latticeHash(h)^uint(NONSTATIONARY+2*THREE_SIGMA+4*CONDITION_STATIONARY_BIRTH+8*SMOOTH_MEAN_NORMALS);
    return feedbackKey(latticeHash(h)^uint(ENSEMBLE_TARGET));
}

void mainImage(out vec4 fragColor,in vec2 fragCoord) {
    ivec2 px=ivec2(fragCoord), size=ivec2(iResolution.xy);
    float strip=floor(0.24*iResolution.y), panelW=0.5*iResolution.x;
    float unit=2.39/min(iResolution.y-strip,panelW);
    vec2 probe=selectedProbe(strip,panelW,unit);
    vec4 scene=sceneKey(size);
    vec4 pick=feedbackKey(latticeHash(floatBitsToUint(probe.x))^floatBitsToUint(probe.y));
    int target=clamp(ENSEMBLE_TARGET,1,1024);
    // Reserve three texels outside the displayed panels for feedback and overlay state.
    if(px.y==1) {
        fragColor=vec4(0);
        if(px.x==0) fragColor=scene;
        if(px.x==1) fragColor=pick;
        if(px.x==2) fragColor=vec4(vec2(0.5)+probe/(unit*vec2(panelW,iResolution.y-strip)),
                                   float(target),strip);
        return;
    }
    if(px.y>1 && fragCoord.y<strip) { fragColor=vec4(0); return; }

    bool valid=iFrame>0 && all(equal(textureSize(iChannel0,0),size));
    if(valid) valid=all(equal(texelFetch(iChannel0,ivec2(0,1),0),scene));
    if(valid && px.y==0) valid=all(equal(texelFetch(iChannel0,ivec2(1,1),0),pick));
    vec4 old=vec4(0);
    if(valid) old=texelFetch(iChannel0,px,0);
    int n=int(old.a);
    bool single=px.y!=0 && fragCoord.x<panelW;
    if(n>=(single?1:target)) { fragColor=old; return; }

    vec3 rd,ex,ey; camera(ENSEMBLE_VIEW_TIME,rd,ex,ey);
    uint realization=sampleRealization(n);
    if(px.y==0) {
        // One histogram row. Every column samples the same flight, each with
        // its own CDF threshold. A miss is T=infinity and stays in the denominator.
        beginFlight(phaseJet(probe).x,realization);
        vec3 ro=flightOrigin(probe,rd,ex,ey);
        vec2 span=processSpan(ro,rd);
        Contact hit=traceShaderball(ro,rd,span,true);
        float normalizedT=hit.t<0.0 ? -1.0 : hit.t/FAR_T;
        float threshold=float(px.x)/max(iResolution.x-1.0,1.0);
        float crossed=(normalizedT>=0.0 && normalizedT<=threshold)?1.0:0.0;
        fragColor=vec4(old.r+crossed,old.g+float(hit.t>=0.0),
                       n==0?normalizedT:old.b,float(n+1));
    } else {
        vec2 plane=panelPoint(fragCoord,strip,panelW,unit);
        vec3 radiance=renderFlight(plane,rd,ex,ey,realization,true);
        fragColor=vec4(old.rgb+(radiance-old.rgb)/float(n+1),float(n+1));
    }
}
#endif
