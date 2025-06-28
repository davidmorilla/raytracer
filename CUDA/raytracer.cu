/*==================[inclusions]============================================*/

#include <omp.h>
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "lib/tinyobj_loader.h"

#include "raytracer.h"
#include "vector.h"
#include <curand_kernel.h>


/*==================[macros]================================================*/
#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)


/*==================[type definitions]======================================*/
/*==================[external function declarations]========================*/
/*==================[internal function declarations]========================*/

__device__ static float mix(float a, float b, float mix);

__device__ static vec3 random_on_unit_sphere();
__device__ static vec3 random_on_hemisphere(vec3, /*curandState*/ uint *state);

static vec3 phong(vec3 color, vec3 light_dir, vec3 normal, vec3 camera_origin, vec3 position, bool in_shadow, float ka, float ks, float kd, float alpha);
static Ray get_camera_ray(const Camera *camera, float u, float v);

static vec3 cast_ray(Ray *ray, Object *scene, size_t nobj, int depth);
static __device__ vec3 trace_path(Ray *ray, Object *scene, size_t nobj, int depth, /*curandState*/ uint* state);

static vec3 reflect(const vec3 In, const vec3 N);
static vec3 refract(const vec3 In, const vec3 N, float iot);

static vec3 checkered_texture(vec3 color, float u, float v, float M);
__global__ void render_pixel(Camera * __restrict__ camera, Object * __restrict__ objects, uint8_t * __restrict__ framebuffer);
static bool intersect(const Ray *ray, Object *objects, size_t n, Hit *hit);

/*==================[external constants]====================================*/
/*==================[internal constants]====================================*/
/*==================[external data]=========================================*/

long long *ray_count;
long long *intersection_test_count;

/*==================[internal data]=========================================*/
/*==================[external function definitions]=========================*/

vec3 calculate_surface_normal(vec3 v0, vec3 v1, vec3 v2)
{ 
  return vec3_normalize(vec3_cross(vec3_sub(v2, v0), vec3_sub(v1, v0))); 
}

void init_camera(Camera *camera, vec3 position, vec3 target, Options *options)
{
  float theta = 60.0 * (PI / 180);
  float h = tan(theta / 2);
  float viewport_height = 2.0 * h;
  float aspect_ratio = (float)options->width / (float)options->height;
  float viewport_width = aspect_ratio * viewport_height;

  vec3 forward = vec3_normalize(vec3_sub(target, position));
  vec3 right = vec3_normalize(vec3_cross((vec3){0, 1, 0}, forward));
  vec3 up = vec3_normalize(vec3_cross(forward, right));

  camera->position = position;
  camera->vertical = vec3_scalar_mult(up, viewport_height);
  camera->horizontal = vec3_scalar_mult(right, viewport_width);

  vec3 half_vertical = vec3_scalar_div(camera->vertical, 2);
  vec3 half_horizontal = vec3_scalar_div(camera->horizontal, 2);

  vec3 llc_old = vec3_sub(
      vec3_sub(camera->position, half_horizontal),
      vec3_sub(half_vertical, (vec3){0, 0, 1}));

  vec3 llc_new = vec3_sub(
      vec3_sub(camera->position, half_horizontal),
      vec3_sub(half_vertical, vec3_scalar_mult(forward, -1)));

  camera->lower_left_corner = llc_new;
}

__device__ bool intersect_sphere(const Ray *ray, vec3 center, float radius, Hit *hit)
{
  
    float Lx = center.x - ray->origin.x;
  float Ly = center.y - ray->origin.y;
  float Lz = center.z - ray->origin.z;

  float tca = Lx * ray->direction.x + Ly * ray->direction.y + Lz * ray->direction.z;
  if (tca < 0) return false;
  
  float d2 = (Lx * Lx + Ly * Ly + Lz * Lz) - tca * tca;
  float radius2 = radius * radius;
  if (d2 > radius2) return false;

  float thc = sqrt(radius2 - d2);
  // solutions for t if the ray intersects
  float t0 = tca - thc;
  float t1 = tca + thc;
  // Ensure t0 is the smaller of the two
  float min_t = (t0 < t1) ? t0 : t1;
  float max_t = (t0 < t1) ? t1 : t0;

  // Prefer min_t, but fall back to max_t if it's negative
  float t = (min_t > EPSILON) ? min_t :
           (max_t > EPSILON) ? max_t : -1.0;

  if (t < 0.0) return false;
  hit->t = t;
  return true;
}

bool intersect_triangle(const Ray *ray, Vertex vertex0, Vertex vertex1, Vertex vertex2, Hit *hit)
{
  intersection_test_count++;

  // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
  vec3 v0, v1, v2;
  v0 = vertex0.pos;
  v1 = vertex1.pos;
  v2 = vertex2.pos;

  vec3 edge1, edge2, h, s, q;
  float a, f, u, v, t;
  edge1 = vec3_sub(v1, v0);
  edge2 = vec3_sub(v2, v0);
  h = vec3_cross(ray->direction, edge2);
  a = vec3_dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return false; // This ray is parallel to this triangle.
  f = 1.0 / a;
  s = vec3_sub(ray->origin, v0);
  u = f * vec3_dot(s, h);
  if (u < 0.0 || u > 1.0)
    return false;
  q = vec3_cross(s, edge1);
  v = f * vec3_dot(ray->direction, q);
  if (v < 0.0 || u + v > 1.0)
    return false;
  // At this stage we can compute t to find out where the intersection point is on the line.
  t = f * vec3_dot(edge2, q);

  if (t > EPSILON)
  {
    hit->t = t;

    vec2 st0 = vertex0.tex;
    vec2 st1 = vertex1.tex;
    vec2 st2 = vertex2.tex;

    vec2 tex = vec2_add(
      vec2_add(
        vec2_scalar_mult(st0, 1 - u - v), 
        vec2_scalar_mult(st1, u)
      ), 
      vec2_scalar_mult(st2, v)
    );

    hit->u = tex.x;
    hit->v = tex.y;
    return true;
  }
  else
  {
    return false;
  }
}

typedef struct {
    float inv_gamma;
    uint width;
    uint height;
    uint samples;
    uint master_seed;
    uint total_n_pixels;
    uint n_objects;
    int depth;
} RenderOptions;

__device__ __constant__ RenderOptions d_opts;
 __host__ __device__ void print_object(const Object *obj, int index) {
    printf("Object[%d]:\n", index);
    printf("  flags: %u\n", obj[index].flags);
    printf("  radius: %f\n", obj[index].radius);
    printf("  center: (%f, %f, %f)\n", obj[index].center.x, obj[index].center.y, obj[index].center.z);
    printf("  color: (%f, %f, %f)\n", obj[index].color.x, obj[index].color.y,obj[index].color.z);
    printf("  emission: (%f, %f, %f)\n", obj[index].emission.x, obj[index].emission.y, obj[index].emission.z);
}


#define CLAMP_FLOAT(x) (fminf(fmaxf((x), 0.0f), 1.0f))
__device__ unsigned int xor_shift(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ float random_float(unsigned int* state) {
    return (float)xor_shift(state) / UINT_MAX;
}
__device__ float random_float_range(float min, float max, unsigned int* state) {
    return min + (max - min) * random_float(state);
}
__global__ void render_sample(Camera* __restrict__ camera, Object* __restrict__ objects, float* __restrict__ accum_buffer) {
    extern __shared__ char shared_mem[];
    
    // Layout shared memory: first objects, then accumulation buffer
    Object* shared_scene = (Object*)shared_mem;
    float* shared_accum = (float*)(shared_mem + d_opts.n_objects * sizeof(Object));
    
    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_id >= d_opts.width * d_opts.height || sample_id >= d_opts.samples) return;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;

    // Load scene into shared memory (once per block)
    for (int i = tid; i < d_opts.n_objects; i += nthreads) {
        shared_scene[i] = objects[i];
    }
    __syncthreads();

    // Initialize shared accumulation per pixel (only by threads where threadIdx.y == 0)
    if (threadIdx.y == 0) {
        shared_accum[threadIdx.x * 3 + 0] = 0.0f;
        shared_accum[threadIdx.x * 3 + 1] = 0.0f;
        shared_accum[threadIdx.x * 3 + 2] = 0.0f;
    }
    __syncthreads();

    // Calculate UV and trace ray
    int x = pixel_id % d_opts.width;
    int y = pixel_id / d_opts.width;

    int global_id = sample_id * d_opts.width * d_opts.height + pixel_id;
    uint state = d_opts.master_seed + global_id;

    float u = (x + random_float(&state)) / ((float)d_opts.width - 1.0f);
    float v = (y + random_float(&state)) / ((float)d_opts.height - 1.0f);

    Ray ray = get_camera_ray(camera, u, v);
    vec3 color = trace_path(&ray, shared_scene, d_opts.n_objects, d_opts.depth, &state);

    // Accumulate in shared memory atomically (multiple samples per pixel processed by different threads in y)
    atomicAdd(&shared_accum[threadIdx.x * 3 + 0], color.x);
    atomicAdd(&shared_accum[threadIdx.x * 3 + 1], color.y);
    atomicAdd(&shared_accum[threadIdx.x * 3 + 2], color.z);

    __syncthreads();

    // After all samples processed, thread with threadIdx.y == 0 writes the pixel accumulation to global memory
    if (threadIdx.y == 0) {
        int index = pixel_id * 3;
        atomicAdd(&accum_buffer[index + 0], shared_accum[threadIdx.x * 3 + 0]);
        atomicAdd(&accum_buffer[index + 1], shared_accum[threadIdx.x * 3 + 1]);
        atomicAdd(&accum_buffer[index + 2], shared_accum[threadIdx.x * 3 + 2]);
    }
}

__global__ void finalize_image(float* __restrict__ accum_buffer, uint8_t* __restrict__ framebuffer) {

    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= d_opts.width * d_opts.height) return;

    int index = pixel_id * 3;
    float inv_samples = 1.0f / d_opts.samples;

    float r = powf(accum_buffer[index + 0] * inv_samples, d_opts.inv_gamma);
    float g = powf(accum_buffer[index + 1] * inv_samples, d_opts.inv_gamma);
    float b = powf(accum_buffer[index + 2] * inv_samples, d_opts.inv_gamma);

    framebuffer[index + 0] = (uint8_t)(255.0f * CLAMP_FLOAT(r));
    framebuffer[index + 1] = (uint8_t)(255.0f * CLAMP_FLOAT(g));
    framebuffer[index + 2] = (uint8_t)(255.0f * CLAMP_FLOAT(b));


}

void render(uint8_t* framebuffer, uint fbuffer_len, Object* objects, unsigned long n_objects, Camera* camera, Options* options) {

    const float gamma = 5.0;
    const float inv_gamma = 1.0f / gamma;
    unsigned int master_seed = time(NULL);
    unsigned int total_n_pixels = options->height * options->width;

    uint8_t* device_framebuffer;
    float* device_accum_buffer;
    Object* device_objects;
    Camera* device_camera;

    RenderOptions render_options = {inv_gamma, options->width, options->height, options->samples, master_seed, total_n_pixels, n_objects, 0};

    CUDA_CHECK(cudaMalloc((void**)&device_framebuffer, fbuffer_len * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&device_accum_buffer, fbuffer_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_objects, n_objects * sizeof(Object)));
    CUDA_CHECK(cudaMalloc((void**)&device_camera, sizeof(Camera)));

    CUDA_CHECK(cudaMemset(device_framebuffer, 0, fbuffer_len * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(device_accum_buffer, 0, fbuffer_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_objects, objects, n_objects * sizeof(Object), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_opts, &render_options, sizeof(RenderOptions)));

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));

    dim3 blockDim(16, 16);
    dim3 gridDim((total_n_pixels + blockDim.x - 1) / blockDim.x, (options->samples + blockDim.y - 1) / blockDim.y);

    int tic = time(NULL);
    int shared_mem_size = n_objects * sizeof(Object) + blockDim.x * 3 * sizeof(float);
    render_sample<<<gridDim, blockDim, shared_mem_size>>>(device_camera, device_objects, device_accum_buffer);
    cudaDeviceSynchronize();
    int toc = time(NULL);
    printf("Render_sample time: %d (%d - %d)\n", toc - tic, toc, tic);

    dim3 finalBlock(256);
    dim3 finalGrid((total_n_pixels + finalBlock.x - 1) / finalBlock.x);
    tic=time(NULL);
    finalize_image<<<finalGrid, finalBlock>>>(device_accum_buffer, device_framebuffer);
    cudaDeviceSynchronize();
    toc=time(NULL);
    printf("Finalize_image time: %d (%d - %d)\n", toc - tic, toc, tic);
    tic = time(NULL);
    CUDA_CHECK(cudaMemcpy(framebuffer, device_framebuffer, fbuffer_len * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    toc = time(NULL);
    printf("cuda memcpy time: %d (%d - %d)\n", toc - tic, toc, tic);

    CUDA_CHECK(cudaFree(device_camera));
    CUDA_CHECK(cudaFree(device_objects));
    CUDA_CHECK(cudaFree(device_framebuffer));
    CUDA_CHECK(cudaFree(device_accum_buffer));
}

/*==================[internal function definitions]=========================*/

float random_double(uint* seed) {return (float)rand_r(seed) / ((float)RAND_MAX + 1); }
float random_range(float min, float max, uint *seed){ return random_double(seed) * (max - min) + min; }
__device__ float d_random_double(curandState *state) {
    return curand_uniform_double(state);  // returns float in (0.0, 1.0]
}

__device__ float d_random_range(float min, float max, curandState *state) {
    return min + (max - min) * d_random_double(state);
}

__device__ vec3 random_on_unit_sphere(uint *state) {
    float z = random_float_range(-1.0f, 1.0f, state);
    float a = random_float_range(0.0f, 2.0f * M_PI, state);

    float r = sqrtf(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);

    return VECTOR(x, y, z);
}

__device__ vec3 random_on_hemisphere(vec3 normal,/*curandState*/ uint *state)
{
  vec3 d = random_on_unit_sphere(state);

  if (vec3_dot(d, normal) < 0)
    return vec3_scalar_mult(d, -1);
  else
    return d;
}

__device__ float mix(float a, float b, float mix) { return b * mix + a * (1 - mix); } 

__device__ vec3 point_at(const Ray *ray, float t) { return vec3_add(ray->origin, vec3_scalar_mult(ray->direction, t)); }

vec3 clamp(const vec3 v) { return (vec3){CLAMP(v.x), CLAMP(v.y), CLAMP(v.z)}; }



void print_v(const char *msg, const vec3 v)
{
  printf("%s: (vec3) { %f, %f, %f }\n", msg, v.x, v.y, v.z);
}

void print_m(const mat4 m)
{
  for (uint i = 0; i < 4; i++)
  {
    for (uint j = 0; j < 4; j++)
    {
      printf(" %6.1f, ", m[i * 4 + j]);
    }
    printf("\n");
  }
}

__device__ vec3 reflect(const vec3 In, const vec3 N)
{
  return vec3_sub(In, vec3_scalar_mult(N, 2 * vec3_dot(In, N)));
}

__device__ vec3 refract(const vec3 In, const vec3 N, float iot)
{
  float cosi = CLAMP_BETWEEN(dot(In, N), -1, 1);
  float etai = 1, etat = iot;
  vec3 n = N;
  if (cosi < 0)
  {
    cosi = -cosi;
  }
  else
  {
    float tmp = etai;
    etai = etat;
    etat = tmp;
    n = vec3_scalar_mult(N, -1);
  }
  float eta = etai / etat;
  float k = 1 - eta * eta * (1 - cosi * cosi);
  return k < 0 ? ZERO_VECTOR : vec3_add(vec3_scalar_mult(In, eta), vec3_scalar_mult(n, eta * cosi - sqrtf(k)));
}

__device__ __forceinline__ Ray get_camera_ray(const Camera *camera, float u, float v)
{
  vec3 direction = vec3_sub(
      camera->position,
      vec3_add(
          camera->lower_left_corner,
          vec3_add(vec3_scalar_mult(camera->horizontal, u), vec3_scalar_mult(camera->vertical, v))));

  return (Ray){camera->position, vec3_normalize(direction)};
}

__device__ vec3 checkered_texture(vec3 color, float u, float v, float M)
{
  float checker = (fmod(u * M, 1.0) > 0.5) ^ (fmod(v * M, 1.0) < 0.5);
  float c = 0.3 * (1 - checker) + 0.7 * checker;
  return vec3_scalar_mult(color, c);
}

__device__ bool intersect(const Ray *ray, Object *objects, size_t n, Hit *hit)
{
  // ray_count++;
  float old_t = hit != NULL ? hit->t : DBL_MAX;
  float min_t = old_t;

  Hit local = {.t = DBL_MAX};
  
  for (uint i = 0; i < n; i++)
  {
   
    if (intersect_sphere(ray, objects[i].center, objects[i].radius, &local) && local.t < min_t)
    {
      min_t = local.t;
      local.object_id = i;
      
    vec3 temp = vec3_add(ray->origin, vec3_scalar_mult(ray->direction, local.t)); // inlined point_at
    vec3 n = vec3_sub(temp, objects[i].center);
    float inv_len = 1.0 / sqrt(n.x * n.x + n.y * n.y + n.z * n.z); // normalized vector length
    n.x *= inv_len;
    n.y *= inv_len;
    n.z *= inv_len;
    local.point = temp;
    local.normal = n;
    local.u = atan2(n.x, n.z) * (1.0 / (2.0 * PI)) + 0.5;
      local.v = local.normal.y * 0.5 + 0.5;
    }
  }

  if (hit != NULL)
  {
    memcpy(hit, &local, sizeof(*hit));
  }

  return min_t < old_t;
}

__device__ vec3 phong(vec3 color, vec3 light_dir, vec3 normal, vec3 camera_origin, vec3 position, bool in_shadow, float ka, float ks, float kd, float alpha)
{
  // ambient
  vec3 ambient = vec3_scalar_mult(color, ka);

  // diffuse
  vec3 diffuse = vec3_scalar_mult(color, kd * MAX(vec3_dot(normal, light_dir), 0.0));

  // specular
  vec3 view_dir = vec3_normalize(vec3_sub(position, camera_origin));
  vec3 reflected = reflect(light_dir, normal);
  vec3 specular = vec3_scalar_mult(color, ks * pow(MAX(vec3_dot(view_dir, reflected), 0.0), alpha));

  return in_shadow ? ZERO_VECTOR : clamp(vec3_add(vec3_add(ambient, diffuse), specular));
}



__device__ vec3 trace_path(Ray *ray, Object *scene, size_t nobj, int depth, /*curandState*/ uint *state)
{
    vec3 radiance = {0, 0, 0};
    vec3 throughput = {1, 1, 1};
  

    while (depth < MAX_DEPTH)
    {
        Hit hit = { .t = DBL_MAX };
        if (!intersect(ray, scene, nobj, &hit)) {
            radiance = vec3_add(radiance, vec3_mult(throughput, BACKGROUND));
            break;
        }

        Object obj = scene[hit.object_id];
        vec3 albedo = obj.color;
        vec3 emission = obj.emission;

        radiance = vec3_add(radiance, vec3_mult(throughput, emission));

        float prob = MAX(albedo.x, MAX(albedo.y, albedo.z));
        if (/*d_random_double*/random_float(state) > prob)
            break;
        albedo = vec3_scalar_mult(albedo, 1.0f / prob);
        throughput = vec3_mult(throughput, albedo);

        // Determine next ray
        Ray next_ray;
        next_ray.origin = hit.point;

        if (obj.flags & M_REFRACTION) {
            float facingratio = -vec3_dot(ray->direction, hit.normal);
            float fresnel = mix(pow(1 - facingratio, 3), 1, 0.1);
            float kr = fresnel;
            float kt = (1 - fresnel);

            Ray refr_ray = { hit.point, vec3_normalize(refract(vec3_scalar_mult(ray->direction, -1), hit.normal, 1.0)) };
            Ray refl_ray = { hit.point, vec3_normalize(reflect(ray->direction, hit.normal)) };

            if (/*d_random_double*/random_float(state) < kr)
                *ray = refl_ray;
            else
                *ray = refr_ray;

        } else if (obj.flags & M_REFLECTION) {
            ray->origin = hit.point;
            ray->direction = reflect(ray->direction, hit.normal);

        } else {
            ray->origin = hit.point;
            ray->direction = random_on_hemisphere(hit.normal, state);
            throughput = vec3_scalar_mult(throughput, vec3_dot(ray->direction, hit.normal));
        }

        ++depth;
    }

    return radiance;
}

__device__ vec3 cast_ray(Ray *ray, Object *objects, size_t nobj, int depth)
{
 // ray_count++;
  Hit hit = {.t = DBL_MAX };

  if (depth > MAX_DEPTH || !intersect(ray, objects, nobj, &hit))
  {
    return BACKGROUND;
  }

  vec3 out_color = ZERO_VECTOR;
  vec3 light_pos = {2, 7, 2};
  vec3 light_color = {1, 1, 1};

  Ray light_ray = {hit.point, vec3_normalize(vec3_sub(light_pos, hit.point))};

  bool in_shadow = intersect(&light_ray, objects, nobj, NULL);
  
  vec3 object_color = objects[hit.object_id].color;
  uint flags = objects[hit.object_id].flags;

  float ka = 0.25;
  float kd = 0.5;
  float ks = 0.8;
  float alpha = 10.0;

  if (flags & M_CHECKERED)
  {
    object_color = checkered_texture(object_color, hit.u, hit.v, 10);
  } 

  vec3 ambient = vec3_scalar_mult(light_color, ka);

  vec3 diffuse = vec3_scalar_mult(light_color, kd * MAX(0.0, vec3_dot(hit.normal, light_ray.direction)));

  vec3 reflected = reflect(light_ray.direction, hit.normal);
  vec3 view_dir = vec3_normalize(vec3_sub(hit.point, ray->origin));
  vec3 specular = vec3_scalar_mult(light_color, ks * pow(MAX(vec3_dot(view_dir, reflected), 0.0), alpha));

  vec3 surface = vec3_mult(
    vec3_add(
      ambient, 
      vec3_scalar_mult(
        vec3_add(specular, diffuse),
        in_shadow ? 0 : 1
      ) 
    ), 
    object_color
  );
  
  vec3 reflection = ZERO_VECTOR, refraction = ZERO_VECTOR;

  float kr = 0, kt = 0;

  if (flags & M_REFLECTION)
  {
    kr = 1.0;
    Ray r = { hit.point, vec3_normalize(reflect(ray->direction, hit.normal)) };
    reflection = cast_ray(&r, objects, nobj, depth + 1);
  }
  
  if (flags & M_REFRACTION)
  {
    float transparency = 0.5;
    float facingratio  = -vec3_dot(ray->direction, hit.normal);
    float fresnel      = mix(pow(1 - facingratio, 3), 1, 0.1);

    kr = fresnel;
    kt = (1 - fresnel) * transparency;

    Ray r = { hit.point, vec3_normalize(refract(ray->direction, hit.normal, 1.0))};
    refraction = cast_ray(&r, objects, nobj, depth + 1);
  }

  out_color = vec3_add(out_color, surface);

  out_color = vec3_add(
    out_color, 
    vec3_add(
      vec3_scalar_mult(reflection, kr),
      vec3_scalar_mult(refraction, kt)
    ) 
  );

  return out_color;
}

/*==================[end of file]===========================================*/
