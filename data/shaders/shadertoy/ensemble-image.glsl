// Shadertoy Image. Set iChannel0 to Buffer A (current frame).
// Renders UI in display RGB; only the accumulated RGB goes through the studio transfer.
precision highp float;
precision highp int;

vec3 ensembleDisplay(vec3 linearColor) {
    return pow(vec3(1.0) - exp(-1.5 * max(linearColor, vec3(0.0))), vec3(1.0 / 2.2));
}

// Small pixel font, row-major, top to bottom. Includes only the labels used here.
float ensembleGlyph(vec2 p, int id) {
    if (p.x < 0.0 || p.y < 0.0 || p.x >= 5.0 || p.y >= 5.0) return 0.0;
    const uint bits[26] = uint[26](
        0xe9d72eu, 0xe210c4u, 0x1f1322eu, 0xf83a0fu, 0x847d29u, 0xf83c3fu, 0xe8bc2eu, 0x22221fu, 0xe8ba2eu, 0xe87a2eu,
        0xf8be2fu, 0x1e0843eu, 0xf8c62fu, 0x1f0bc3fu, 0x10bc3fu, 0x1e8f43eu, 0x118fe31u, 0x1f2109fu, 0x1f08421u, 0x118d771u,
        0x11cd671u, 0xf8383eu, 0x42109fu, 0x111110u, 0xf83e0u, 0x0u);
    return float((bits[id] >> uint(5 * int(p.y) + int(p.x))) & 1u);
}

// IDs: 0..9 B C D E F G H I L M N S T / = space
float ensembleLabel(vec2 p, int word) {
    if (p.x < 0.0 || p.x >= 48.0 || p.y < 0.0 || p.y >= 5.0) return 0.0;
    const int single[8] = int[8](21,17,20,15,18,13,25,25);
    const int ensemble[8] = int[8](13,20,21,13,19,10,18,13);
    const int cdf[8] = int[8](16,17,22,25,11,12,14,25);
    int i = int(p.x / 6.0);
    int id = word == 0 ? single[i] : (word == 1 ? ensemble[i] : cdf[i]);
    return ensembleGlyph(vec2(p.x - 6.0 * float(i), p.y), id);
}

int ensembleDigits(int n) {
    if (n >= 1000) return 4;
    if (n >= 100) return 3;
    if (n >= 10) return 2;
    return 1;
}

float ensembleNumber(vec2 p, int n) {
    int len = ensembleDigits(n);
    if (p.x < 0.0 || p.x >= float(len) * 6.0 || p.y < 0.0 || p.y >= 5.0) return 0.0;
    int i = int(p.x / 6.0);
    const int powers[4] = int[4](1,10,100,1000);
    int digit = (n / powers[len - 1 - i]) % 10;
    return ensembleGlyph(vec2(p.x - 6.0 * float(i), p.y), digit);
}

float ensembleCount(vec2 p, int n) {
    return max(ensembleGlyph(p,20), max(ensembleGlyph(p - vec2(6,0),24),
                                        ensembleNumber(p - vec2(12,0),n)));
}

float ensembleLine(float d, float width) {
    return 1.0 - smoothstep(max(width - 0.65,0.0),width + 0.65, d);
}

float ensembleVertical(vec2 p, float x, float y0, float y1) {
    float dy = max(max(min(y0,y1)-p.y, p.y-max(y0,y1)), 0.0);
    return ensembleLine(length(vec2(x-p.x, dy)), 1.0);
}

vec3 ensembleCDF(vec2 p, float strip, float scale, int width) {
    vec3 color = mix(vec3(0.045,0.064,0.085), vec3(0.068,0.095,0.115),
                     clamp(p.y/max(strip,1.0),0.0,1.0));
    vec3 ink = vec3(0.82,0.89,0.91);
    float left = 15.0 * scale, right = max(left+1.0, iResolution.x-9.0*scale);
    float low = 7.0 * scale, high = max(low+1.0, strip-15.0*scale);
    float time = clamp((p.x-left)/(right-left),0.0,1.0);
    int last = max(width-1,0);
    int col = min(int(floor(time*float(last))),last);
    vec4 bin = texelFetch(iChannel0,ivec2(col,0),0);
    int n = int(max(0.0,bin.a)+0.5);

    if (p.x >= left && p.x <= right && p.y >= low-1.0 && p.y <= high+1.0) {
        // The bins store unconditional counts. A miss stays in the denominator.
        float grid = min(abs(p.y-low),min(abs(p.y-high), abs(p.y-0.5*(low+high))));
        float gridX = abs(fract(time*8.0+0.5)-0.5)*(right-left)/8.0;
        color = mix(color,vec3(0.25,0.34,0.37),
                    max(0.58*ensembleLine(grid,0.6),0.26*ensembleLine(gridX,0.5)));
        if (n > 0) {
            float prev = texelFetch(iChannel0,ivec2(max(col-1,0),0),0).r;
            float next = texelFetch(iChannel0,ivec2(min(col+1,width-1),0),0).r;
            float range = (high-low)/float(n);
            float y = low + bin.r*range;
            float binWidth = (right-left)/float(max(last,1));
            float x0 = left + float(col)*binWidth;
            float line = ensembleLine(abs(p.y-y),1.0);
            line = max(line,ensembleVertical(p,x0,low+prev*range,y));
            line = max(line,ensembleVertical(p,x0+binWidth,y,low+next*range));
            color = mix(color,vec3(1.0,0.76,0.40),line);
            if (bin.b >= 0.0 && bin.b <= 1.0) {
                float hitX = mix(left,right,bin.b);
                float dash = 1.0-smoothstep(2.0,3.4,abs(mod(p.y-low+4.0,8.0)-4.0));
                float marker = ensembleLine(length(p-vec2(hitX,low)),2.45);
                color = mix(color,vec3(1.0,0.46,0.34),
                            max(marker,0.63*dash*ensembleLine(abs(p.x-hitX),0.9)));
            }
        }
    }

    vec2 header = vec2(p.x/scale-7.0,(strip-p.y)/scale-4.0);
    float labels = ensembleLabel(header,2);
    float countWidth = float(2+ensembleDigits(n))*6.0;
    vec2 countPos = vec2(p.x/scale-(iResolution.x/scale-9.0-countWidth),header.y);
    labels = max(labels,ensembleCount(countPos,n));
    vec2 labelX = vec2(p.x/scale-5.0,0.0);
    labels = max(labels,ensembleGlyph(vec2(labelX.x,(low-p.y)/scale+2.5),0));
    labels = max(labels,ensembleGlyph(vec2(labelX.x,(high-p.y)/scale+2.5),1));
    return mix(color,ink,labels);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 bufferSize = textureSize(iChannel0,0);
    vec4 meta = texelFetch(iChannel0,min(ivec2(2,1),bufferSize-1),0);
    float strip = meta.w > 0.0 ? meta.w : floor(0.24*iResolution.y);
    float scale = max(1.0,floor(iResolution.y/230.0));
    vec3 rgb;
    if (fragCoord.y < strip) {
        rgb = ensembleCDF(fragCoord,strip,scale,max(bufferSize.x,1));
    } else {
        ivec2 coord = clamp(ivec2(fragCoord),ivec2(0),bufferSize-1);
        vec4 stored = texelFetch(iChannel0,coord,0);
        rgb = ensembleDisplay(stored.rgb);
        int n = int(max(stored.a,0.0)+0.5);
        float panelWidth = 0.5*iResolution.x;
        bool multi = fragCoord.x >= panelWidth;
        vec2 panelPixel = vec2(fragCoord.x - (multi ? panelWidth : 0.0),fragCoord.y-strip);
        vec2 panelSize = vec2(panelWidth,iResolution.y-strip);
        float radius = length(panelPixel-meta.xy*panelSize);
        float ring = ensembleLine(abs(radius-max(4.0,2.6*scale)),0.9);
        rgb = mix(rgb,vec3(1.0,0.76,0.46),0.85*ring);
        float seam = ensembleLine(abs(fragCoord.x-panelWidth),0.75);
        float edge = ensembleLine(abs(fragCoord.y-strip),0.8);
        rgb = mix(rgb,vec3(0.17,0.23,0.25),max(seam,edge));

        vec2 textPos = vec2(panelPixel.x,iResolution.y-fragCoord.y)/scale - 9.0;
        float label = ensembleLabel(textPos,multi ? 1 : 0);
        if (multi) {
            vec2 np = textPos - vec2(55.0,0.0);
            if (panelWidth/scale < 120.0) np = textPos - vec2(0.0,8.0);
            label = max(label,ensembleCount(np,n));
            int target = int(max(meta.z,0.0)+0.5);
            if (target > 0 && panelWidth/scale >= 140.0) {
                vec2 rest = np - vec2(float(2+ensembleDigits(n))*6.0,0.0);
                label = max(label,max(ensembleGlyph(rest,23),ensembleNumber(rest-vec2(6,0),target)));
            }
        }
        rgb = mix(rgb,vec3(0.91,0.96,0.94),label);
    }
    fragColor = vec4(rgb,1.0);
}
