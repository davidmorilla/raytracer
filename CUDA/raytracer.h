#ifndef RAYTRACER_H
#define RAYTRACER_H

/*==================[inclusions]============================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <tgmath.h>
#include <time.h>
#include <float.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "curand_kernel.h"
#include "vector.h"
#include <cuda_runtime.h>
/*==================[macros]================================================*/

#ifndef PI
#define PI 3.14159265359
#endif
#define EPSILON 1e-8
#define MAX_DEPTH 5
#define MONTE_CARLO_SAMPLES 1
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x) (MAX(0, MIN(x, 1)))
#define CLAMP_BETWEEN(x, min_v, max_v) (MAX(min_v, MIN(max_v, 1)))
#define ABS(x) ((x < 0) ? (-x) : (x))
#define VECTOR(x, y, z) ((vec3) {(x), (y), (z)})
#define RGB(r, g, b) (VECTOR((r) / 255.0, (g) / 255.0, (b) / 255.0))
#define EQ(a, b) (ABS((a) - (b)) < EPSILON)


#define RAY(o, d) ((Ray) { .origin = o, .direction = d })

#define RED       RGB(255, 0, 0)
#define GREEN     RGB(0, 192, 48)
#define BLUE      RGB(0, 0, 255)
#define WHITE     RGB(255, 255, 255)
#define BLACK     RGB(0, 0, 0)
//#define BACKGROUND RGB(0, 0x80, 0x80)

#define BACKGROUND RGB(10, 10, 10)

#define ZERO_VECTOR RGB(0, 0, 0)
#define ONE_VECTOR (VECTOR(1.0, 1.0, 1.0))
#define RANDOM_COLOR(seed) \
  (vec3) { random_double(seed), random_double(seed), random_double(seed) }

#define M_DEFAULT           ((uint)1 << 1)
#define M_REFLECTION        ((uint)1 << 2)
#define M_REFRACTION        ((uint)1 << 3)
#define M_CHECKERED         ((uint)1 << 4)

/*==================[type definitions]======================================*/

typedef uint32_t uint;
typedef struct { vec3 pos; vec2 tex; } Vertex;
typedef struct { vec3 origin, direction; } Ray;

typedef struct
{
  uint flags;
  vec3 color, emission;
  float ka, ks, kd;
} Material;

typedef struct
{
  vec3 center;
  float radius;
} Sphere;

typedef struct
{
  size_t num_triangles;
  Vertex *vertices;
} TriangleMesh;

typedef union
{
  TriangleMesh *mesh;
  Sphere *sphere;
} Geometry;

typedef enum
{
  GEOMETRY_SPHERE,
  GEOMETRY_MESH,
} GeometryType;

typedef struct __align__(64)
{
  uint flags;
  float radius;
  vec3 center;
  vec3 color;
  vec3 emission;
} Object;

typedef struct
{
  float t, u, v;
  vec3 point;
  vec3 normal;
  uint object_id;
} Hit;

typedef struct
{
  vec3 position, horizontal, vertical, lower_left_corner;
} Camera;

typedef struct
{
  vec3 background;
  char *result, *obj;
  int width, height, samples;
} Options;

/*==================[external function declarations]========================*/

float random_double(unsigned int* seed);
float random_range(float, float, uint* seed);
float d_random_double(curandState *state);
float d_random_range(float min, float max, curandState *state);
vec3 point_at(const Ray *ray, float t);

vec3 calculate_surface_normal(vec3 v0, vec3 v1, vec3 v2);

bool intersect_sphere(const Ray *ray, vec3 center, float radius, Hit *hit);
bool intersect_triangle(const Ray *ray, Vertex vertex0, Vertex vertex1, Vertex vertex2, Hit *hit);

#if 0
mat4 scale(vec3);
mat4 rotate(vec3);
mat4 translate(vec3);
#endif

void print_v(const char* msg, const vec3 v);
void print_m(const mat4 m);

void init_camera(Camera *camera, vec3 position, vec3 target, Options *options);

void render(uint8_t *framebuffer, uint buffer_len,  Object *objects, size_t n_objects, Camera *camera, Options *options);

bool load_obj(const char *filename, TriangleMesh *mesh);

/*==================[external constants]====================================*/
/*==================[external data]=========================================*/

extern long long *ray_count;
extern long long *intersection_test_count;

/*==================[end of file]===========================================*/

#endif /* RAYTRACER_H */
