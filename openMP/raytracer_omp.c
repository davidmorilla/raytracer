/*==================[inclusions]============================================*/

#include <omp.h>
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "lib/tinyobj_loader.h"

#include "raytracer_omp.h"
#include "vector.h"

/*==================[macros]================================================*/
/*==================[type definitions]======================================*/
/*==================[external function declarations]========================*/
/*==================[internal function declarations]========================*/

static double mix(double a, double b, double mix);

static vec3 random_on_unit_sphere();
static vec3 random_on_hemisphere(vec3, int *seed);

static vec3 phong(vec3 color, vec3 light_dir, vec3 normal, vec3 camera_origin, vec3 position, bool in_shadow, double ka, double ks, double kd, double alpha);
static Ray get_camera_ray(const Camera *camera, double u, double v);

static vec3 cast_ray(Ray *ray, Object *scene, size_t nobj, int depth);
static vec3 trace_path(Ray *ray, Object *scene, size_t nobj, int depth, int* seed);

static vec3 reflect(const vec3 In, const vec3 N);
static vec3 refract(const vec3 In, const vec3 N, double iot);

static vec3 checkered_texture(vec3 color, double u, double v, double M);

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
  double theta = 60.0 * (PI / 180);
  double h = tan(theta / 2);
  double viewport_height = 2.0 * h;
  double aspect_ratio = (double)options->width / (double)options->height;
  double viewport_width = aspect_ratio * viewport_height;

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

bool intersect_sphere(const Ray *ray, vec3 center, double radius, Hit *hit)
{
    
 //intersection_test_count++;
  intersection_test_count[omp_get_thread_num()]+=1; 
  double Lx = center.x - ray->origin.x;
  double Ly = center.y - ray->origin.y;
  double Lz = center.z - ray->origin.z;

  double tca = Lx * ray->direction.x + Ly * ray->direction.y + Lz * ray->direction.z;
  if (tca < 0) return false;
  
  double d2 = (Lx * Lx + Ly * Ly + Lz * Lz) - tca * tca;
  double radius2 = radius * radius;
  if (d2 > radius2) return false;

  double thc = sqrt(radius2 - d2);
  // solutions for t if the ray intersects
  double t0 = tca - thc;
  double t1 = tca + thc;
  // Ensure t0 is the smaller of the two
  double min_t = (t0 < t1) ? t0 : t1;
  double max_t = (t0 < t1) ? t1 : t0;

  // Prefer min_t, but fall back to max_t if it's negative
  double t = (min_t > EPSILON) ? min_t :
           (max_t > EPSILON) ? max_t : -1.0;

  if (t < 0.0) return false;
  hit->t = t;
  return true;
  
}

bool intersect_triangle(const Ray *ray, Vertex vertex0, Vertex vertex1, Vertex vertex2, Hit *hit)
{
  //intersection_test_count++;

  // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
  vec3 v0, v1, v2;
  v0 = vertex0.pos;
  v1 = vertex1.pos;
  v2 = vertex2.pos;

  vec3 edge1, edge2, h, s, q;
  double a, f, u, v, t;
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

void render(uint8_t *framebuffer, Object *objects, size_t n_objects, Camera *camera, Options *options)
{
    const double gamma = 5.0;
    const int str_len = 40;
    const char* done = "========================================";
    const char* todo = "----------------------------------------";
    unsigned int master_seed = time(NULL);
    int tic = clock();

    #pragma omp parallel
    {
          if (omp_get_thread_num() == 0) {
            // Solo el hilo 0 ejecuta esto
            ray_count = calloc(omp_get_num_threads(), sizeof(long long));
            intersection_test_count = calloc(omp_get_num_threads(), sizeof(long long));
        }

        // Todos los hilos esperan aquí hasta que hilo 0 termine
        #pragma omp barrier
        int thread_id = omp_get_thread_num();
        int seed = master_seed + thread_id * 9973;  // unique corelated seed per thread
        
        #pragma omp for schedule(dynamic)
        for (uint i = 0; i < options->height*options->width; i++)
        {




          uint x= i%options->width, y= i/options->width;
            // progress output (only by thread 0)
            if (thread_id==0 && x == 0)
            {
                double percentage = 100.0 * y / (double)(options->height);
                int p = (percentage / 100.0) * str_len;
                printf("[%.*s%.*s] %0.02f %%\t[%d seconds]\n", 
                    p, done, str_len - p, todo, 
                    percentage, (int)((clock() - tic)/(omp_get_num_threads()*CLOCKS_PER_SEC)));
            }
        
                REAL X = 0.0, Y = 0.0, Z = 0.0;

                for (uint s = 0; s < options->samples; s++)
                {
                    double u = (double)(x + random_double(&seed)) / ((double)options->width - 1.0);
                    double v = (double)(y + random_double(&seed)) / ((double)options->height - 1.0);

                    Ray ray = get_camera_ray(camera, u, v);
                    vec3 sample_pixel = trace_path(&ray, objects, n_objects, 0, &seed);
                    
                    X += sample_pixel.x;
                    Y += sample_pixel.y;
                    Z += sample_pixel.z;
                }

                vec3 pixel = vec3_scalar_mult(VECTOR(X, Y, Z), 1.0 / (double)options->samples);
                float inv_gamma = 1.0f / gamma;
                uint a = (y * options->width + x) * 3;
                framebuffer[a + 0] = (uint8_t)(255.0 * CLAMP(pow(pixel.x, inv_gamma)));
                framebuffer[a + 1] = (uint8_t)(255.0 * CLAMP(pow(pixel.y, inv_gamma)));
                framebuffer[a + 2] = (uint8_t)(255.0 * CLAMP(pow(pixel.z, inv_gamma)));
           
       
       
       
       
              }
      #pragma omp critical
      {
        if (thread_id!=0){
          ray_count[0]+=ray_count[thread_id];
          intersection_test_count[0]+=intersection_test_count[thread_id];
        }
      }
    }
}

/*==================[internal function definitions]=========================*/

double random_double(int* seed) {return (double)rand_r(seed) / ((double)RAND_MAX + 1); }

double random_range(double min, double max, int *seed){ return random_double(seed) * (max - min) + min; }

vec3 random_on_unit_sphere(int *seed)
{
  vec3 p;
  double d = 100000;
  int loop_counter = 0;

  do {
    assert(++loop_counter < 100);
    p = VECTOR(random_range(-1, 1, seed), random_range(-1, 1, seed), random_range(-1, 1, seed));
  } while(vec3_length(p) > 1);

  return vec3_normalize(p);
}

vec3 random_on_hemisphere(vec3 normal, int*seed)
{
  vec3 d = random_on_unit_sphere(seed);

  if (vec3_dot(d, normal) < 0)
    return vec3_scalar_mult(d, -1);
  else
    return d;
}

double mix(double a, double b, double mix) { return b * mix + a * (1 - mix); } 

vec3 point_at(const Ray *ray, double t) { return vec3_add(ray->origin, vec3_scalar_mult(ray->direction, t)); }

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

vec3 reflect(const vec3 In, const vec3 N)
{
  return vec3_sub(In, vec3_scalar_mult(N, 2 * vec3_dot(In, N)));
}

vec3 refract(const vec3 In, const vec3 N, double iot)
{
  double cosi = CLAMP_BETWEEN(dot(In, N), -1, 1);
  double etai = 1, etat = iot;
  vec3 n = N;
  if (cosi < 0)
  {
    cosi = -cosi;
  }
  else
  {
    double tmp = etai;
    etai = etat;
    etat = tmp;
    n = vec3_scalar_mult(N, -1);
  }
  double eta = etai / etat;
  double k = 1 - eta * eta * (1 - cosi * cosi);
  return k < 0 ? ZERO_VECTOR : vec3_add(vec3_scalar_mult(In, eta), vec3_scalar_mult(n, eta * cosi - sqrtf(k)));
}

Ray get_camera_ray(const Camera *camera, double u, double v)
{
  vec3 direction = vec3_sub(
      camera->position,
      vec3_add(
          camera->lower_left_corner,
          vec3_add(vec3_scalar_mult(camera->horizontal, u), vec3_scalar_mult(camera->vertical, v))));

  return (Ray){camera->position, vec3_normalize(direction)};
}

vec3 checkered_texture(vec3 color, double u, double v, double M)
{
  double checker = (fmod(u * M, 1.0) > 0.5) ^ (fmod(v * M, 1.0) < 0.5);
  double c = 0.3 * (1 - checker) + 0.7 * checker;
  return vec3_scalar_mult(color, c);
}

bool intersect(const Ray *ray, Object *objects, size_t n, Hit *hit)
{
  // ray_count++;
  double old_t = hit != NULL ? hit->t : DBL_MAX;
  double min_t = old_t;

  Hit local = {.t = DBL_MAX};
  
  for (uint i = 0; i < n; i++)
  {
    if (intersect_sphere(ray, objects[i].center, objects[i].radius, &local) && local.t < min_t)
    {
          
    min_t = local.t;
    local.object_id = i;

    vec3 temp = vec3_add(ray->origin, vec3_scalar_mult(ray->direction, local.t)); // inlined point_at
    vec3 n = vec3_sub(temp, objects[i].center);
    double inv_len = 1.0 / sqrt(n.x * n.x + n.y * n.y + n.z * n.z); // normalized vector length
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

vec3 phong(vec3 color, vec3 light_dir, vec3 normal, vec3 camera_origin, vec3 position, bool in_shadow, double ka, double ks, double kd, double alpha)
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

vec3 trace_path(Ray *ray, Object *objects, size_t nobj, int depth, int *seed)
{

  ray_count[omp_get_thread_num()]++;
  Hit hit = { .t = DBL_MAX };

  if (depth > MAX_DEPTH || !intersect(ray, objects, nobj, &hit))
  {
    return BACKGROUND;
  }

  vec3 radiance;
  vec3 albedo       = objects[hit.object_id].color;
  vec3 emission     = objects[hit.object_id].emission;

  /* russian roulette */
  double prob = MAX(albedo.x, MAX(albedo.y, albedo.z));

  if (random_double(seed) < prob)
    albedo = vec3_scalar_mult(albedo, 1 / prob);
  else
    return emission;

  uint flags = objects[hit.object_id].flags;

  if (flags & M_CHECKERED)
  {
    albedo = checkered_texture(albedo, hit.u, hit.v, 100000);
  }

  Ray R;
  R.origin = hit.point;

  if (flags & M_REFRACTION)
  {
    double transparency = 1.0;
    double facingratio  = -vec3_dot(ray->direction, hit.normal);
    double fresnel      = mix(pow(1 - facingratio, 3), 1, 0.1);
    double kr           = fresnel;
    double kt           = (1 - fresnel) * transparency;

    R.direction = vec3_normalize(refract(vec3_scalar_mult(ray->direction, -1), hit.normal, 1.0));
    vec3 refraction = trace_path(&R, objects, nobj, depth + 1, seed);
  
    R.direction = vec3_normalize(reflect(vec3_scalar_mult(ray->direction, 1), hit.normal));
    vec3 reflection = trace_path(&R, objects, nobj, depth + 1, seed);
   
    radiance = vec3_add(vec3_scalar_mult(refraction, kt), vec3_scalar_mult(reflection, kr));

  }
  else if(flags & M_REFLECTION)
  {
    R.direction = reflect(ray->direction, hit.normal);
    radiance =  trace_path(&R, objects, nobj, depth + 1, seed);
    
  }
  else 
  {
    R.direction = random_on_hemisphere(hit.normal, seed);
    //double cos_theta = -dot(ray->direction, hit.normal);
    double cos_theta = vec3_dot(R.direction, hit.normal);

    //     radiance =  vec3_scalar_mult(trace_path(&R, objects, nobj, depth + 1, seed), cos_theta);
    vec3 traced = trace_path(&R, objects, nobj, depth + 1, seed);
    radiance.x = traced.x * cos_theta;
    radiance.y = traced.y * cos_theta;
    radiance.z = traced.z * cos_theta;

   
  }
 
  return vec3_add(emission, vec3_mult(albedo, radiance));
}

vec3 cast_ray(Ray *ray, Object *objects, size_t nobj, int depth)
{
  //ray_count++;
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

  double ka = 0.25;
  double kd = 0.5;
  double ks = 0.8;
  double alpha = 10.0;

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

  double kr = 0, kt = 0;

  if (flags & M_REFLECTION)
  {
    kr = 1.0;
    Ray r = { hit.point, vec3_normalize(reflect(ray->direction, hit.normal)) };
    reflection = cast_ray(&r, objects, nobj, depth + 1);
  }
  
  if (flags & M_REFRACTION)
  {
    double transparency = 0.5;
    double facingratio  = -vec3_dot(ray->direction, hit.normal);
    double fresnel      = mix(pow(1 - facingratio, 3), 1, 0.1);

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
