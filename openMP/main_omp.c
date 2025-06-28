#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <tgmath.h>
#include <time.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#include "raytracer_omp.h"
#include "vector.h"

#define SPHERE(x, y, z, r) \
    .center = { (x), (y) + (r), (z) },\
    .radius = (r),\

#define N_SPHERES (25)

Options options = {
    .width = 320,
    .height = 180,
    .samples = 50,
    .result = "result.png",
    .obj = "assets/cube.obj",
};

uint8_t *framebuffer = NULL;

extern long long *ray_count;
extern long long *intersection_test_count;

void write_image(int signal)
{
    if (framebuffer != NULL)
    {
        if (stbi_write_png(options.result, options.width, options.height, 3, framebuffer, options.width * 3) == 0)
            exit(EXIT_FAILURE);
        else
            printf("done.\n");

        free(framebuffer);
    }
}

bool collision(vec3 center0, double radius0, vec3 center1, double radius1, bool debug)
{
    if(debug){
        printf("center0: {%f, %f, %f} radius0: {%f}\ncenter1: {%f, %f, %f} radius1: {%f}\nlength: {%f} < {%f}", center0.x, center0.y, center0.z, radius0, center1.x, center1.y, center1.z, radius1, vec3_length(vec3_sub(center0, center1)), radius0 + radius1);
    }
    return vec3_length(vec3_sub(center0, center1)) < (radius0 + radius1);
}

void test_collision()
{


    Object o_1 = {.radius = 3, .center = {0,0,0}};
    Object o_2 = {.radius = 3, .center = {6,0,0}};

    printf("collision = %d\n", collision(o_1.center, o_1.radius, o_2.center, o_2.radius, false));
}

int generate_random_spheres(Object *spheres, int num_spheres, vec3 box_min, vec3 box_max)
{
    const int max_iterations = 100000000;

    const double min_radius = 2.00, max_radius = 7.00;
    int seed= time(NULL);
    int iterations = 0;
    int spheres_found = 0;
    const double padding = 0.5;

    while(spheres_found < num_spheres)
    {
        assert(iterations++ < max_iterations);

        double radius = random_range(min_radius, max_radius, &seed);
        double diameter = 2*radius;
        vec3 vr = { diameter, diameter, diameter };

        vec3 min = vec3_add(box_min, vr);
        vec3 max = vec3_sub(box_max, vr);

        vec3 center = {
            random_range(min.x, max.x, &seed),
            random_range(min.y, max.y, &seed),
            random_range(min.z, max.z, &seed)
        };

        bool coll = false;
        for (int i = 0; i < spheres_found; i++)
        {
            coll = collision(spheres[i].center, spheres[i].radius, center, radius,false);
            if (coll)
                break;
        }

        if (!coll)
        {
         
            spheres_found++;

            uint flags = M_DEFAULT;
            vec3 color = WHITE;
            vec3 emission = BLACK;

            double r = random_double(&seed);
            if (r < 0.2)
            {
                emission = RANDOM_COLOR(&seed);
            }
            else 
            {
                if (r > 0.8)
                    flags = M_REFRACTION;
                else if(r > 0.4)
                    flags = M_REFLECTION;
            }

            if (vec3_length(emission) > 0)
            {
                //printf("[%d] = { .center = { %f, %f, %f }, .radius = %f, .emission = { %f, %f, %f } },\n", spheres_found, center.x, center.y, center.z, radius, emission.x, emission.y, emission.z);
                printf("[%d] = { .center = { %f, %f, %f }, .radius = %f },\n", spheres_found, center.x, center.y, center.z, radius);
            }

            //center = (vec3){-8.053048, 13.375004, -5.639876};

            spheres[spheres_found-1] = (Object) {
                .center = center,
                .radius = radius,
                .flags = flags,
                .color = color,
                .emission = emission
            };
        }
    }
    return spheres_found;
}

void apply_matrix(TriangleMesh* mesh, mat4 matrix)
{
   
    for (uint i = 0; i < mesh->num_triangles * 3; i++)
    {
        mesh->vertices[i].pos = mat4_vector_mult(matrix, mesh->vertices[i].pos);
    }
}

void parse_options(int argc, char **argv, Options *options)
{
    uint optind;
    for (optind = 1; optind < argc; optind++)
    {
        switch (argv[optind][1])
        {
        case 'h':
            options->height = atoi(argv[optind + 1]);
            break;
        case 'w':
            options->width = atoi(argv[optind + 1]);
            break;
        case 's':
            options->samples = atoi(argv[optind + 1]);
            break;
        case 'o':
            options->result = argv[optind + 1];
            break;

        default:
            break;
        }
    }
    argv += optind;
}

int main(int argc, char **argv)
{
    unsigned int seed;
    seed = (unsigned)time(NULL);

    printf("seed = %d\n", seed);
    srand(seed);
    if (argc <= 1)
    {
        fprintf(stderr, "Usage: %s -w <width> -h <height> -s <samples per pixel> -o <filename>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    parse_options(argc, argv, &options);

    vec3 pos = {0, 0, 0};
    vec3 size = {1, 1, 1.5};

    const TriangleMesh cube = {
        .num_triangles = 2,
        .vertices = (Vertex[]){
            {{ -0.5f, +0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ +0.5f, +0.5f, -0.5f },{ 1.0f, 1.0f }},
            {{ +0.5f, +0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ +0.5f, +0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ -0.5f, +0.5f, +0.5f },{ 0.0f, 0.0f }},
            {{ -0.5f, +0.5f, -0.5f },{ 0.0f, 1.0f }},

            {{ -0.5f, -0.5f, -0.5f },{ 0.0f, 0.0f }},
            {{ +0.5f, -0.5f, -0.5f },{ 1.0f, 0.0f }},
            {{ +0.5f, +0.5f, -0.5f },{ 1.0f, 1.0f }},
            {{ +0.5f, +0.5f, -0.5f },{ 1.0f, 1.0f }},
            {{ -0.5f, +0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ -0.5f, -0.5f, -0.5f },{ 0.0f, 0.0f }},

            {{ -0.5f, -0.5f, +0.5f },{ 0.0f, 0.0f }},
            {{ +0.5f, -0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ +0.5f, +0.5f, +0.5f },{ 1.0f, 1.0f }},
            {{ +0.5f, +0.5f, +0.5f },{ 1.0f, 1.0f }},
            {{ -0.5f, +0.5f, +0.5f },{ 0.0f, 1.0f }},
            {{ -0.5f, -0.5f, +0.5f },{ 0.0f, 0.0f }},
            {{ -0.5f, +0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ -0.5f, +0.5f, -0.5f },{ 1.0f, 1.0f }},
            {{ -0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ -0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ -0.5f, -0.5f, +0.5f },{ 0.0f, 0.0f }},
            {{ -0.5f, +0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ +0.5f, +0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ +0.5f, +0.5f, -0.5f },{ 1.0f, 1.0f }},
            {{ +0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ +0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ +0.5f, -0.5f, +0.5f },{ 0.0f, 0.0f }},
            {{ +0.5f, +0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ -0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f }},
            {{ +0.5f, -0.5f, -0.5f },{ 1.0f, 1.0f }},
            {{ +0.5f, -0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ +0.5f, -0.5f, +0.5f },{ 1.0f, 0.0f }},
            {{ -0.5f, -0.5f, +0.5f },{ 0.0f, 0.0f }},
            {{ -0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f }},
        }
    };

    const double aspect_ratio = (double)options.width / (double)options.height;
    const double room_depth = 30;
    const double room_height = 20;
    const double room_width = room_height * aspect_ratio;
    const double radius = 10000;
    const  vec3 wall_color = VECTOR(0.75, 0.75, 0.75);
    const double light_radius = 15;
    const double y = -room_height;

    uint lighting = M_DEFAULT;

    Object pre_scene[] = { 
     /* walls */
        { // floor
            .color = wall_color, 
            .emission = BLACK,
            .flags = lighting,
            .center =  {0, -radius - room_height, 0},
            .radius = radius,
        },
        { // back wall
            .color = wall_color, 
            .emission = BLACK,
            .flags = lighting,
            .center = {0, 0, -radius - room_depth}, 
            .radius = radius,
        },
        { // left wall
            .color = VECTOR(0.25, 0.75, 0.25), 
            .emission = BLACK,
            .flags = lighting,
            .center =  {-radius - room_width, 0, 0}, 
            .radius = radius,
        },
        { // right wall
            .color = VECTOR(0.75, 0.25, 0.25), 
            .emission = BLACK,
            .flags = lighting,
            .center =  {radius + room_width, 0, 0}, 
            .radius = radius,
        },
        { // ceiling
            .color = wall_color, 
            .emission = BLACK,
            .flags = lighting,
            .center = {0, radius + room_height, 0}, 
            .radius = radius,
        },
        { // front wall
            .color = wall_color, 
            .emission =BLACK,
            .flags = lighting,
            .center = {0, 0, radius + room_depth * 2}, 
            .radius = radius,
        },
        
 /* light */
        {
            .color = WHITE, 
            .flags = M_DEFAULT,
            .emission = RGB(0x00 * 15, 0x32 * 15 -100, 0xA0 * 15-300), //VECTOR(1,1,1), //RGB(0x00 * 15, 0x32 * 15, 0xA0 * 15),
            .center = {0, room_height + light_radius * 0.9, 0}, 
            .radius = light_radius,
        },
        {
            .color = WHITE, 
            .flags = M_DEFAULT,
            .emission = RGB(0xD0, 0x00, 0x70),
            SPHERE(2, y, 12, 3)
        }, 

};
    Object* spheres = malloc(sizeof(Object)*10);
    if (spheres == NULL)
    {
        fprintf(stderr, "could not allocate spheres\n");
        exit(EXIT_FAILURE);
    }
    int num_spheres = generate_random_spheres(spheres, 10, VECTOR(-25,-18,15), VECTOR(25,18, 27));
    printf("num_spheres= %d\n", num_spheres);
    Object* scene = malloc(sizeof(Object)* (num_spheres + 8));
    if (scene == NULL)
    {
        fprintf(stderr, "could not allocate scene\n");
        exit(EXIT_FAILURE);
    }
    memcpy(scene,pre_scene,sizeof(Object)*8);
    memcpy(scene + 8, spheres, sizeof(Object)*num_spheres);
    size_t buff_len = sizeof(*framebuffer) * options.width * options.height * 3;
    framebuffer = malloc(buff_len);
    if (framebuffer == NULL)
    {
        fprintf(stderr, "could not allocate framebuffer\n");
        exit(EXIT_FAILURE);
    }

    memset(framebuffer, 0x0,buff_len);
    signal(SIGINT, &write_image);

    Camera camera;
    init_camera(&camera, VECTOR(0.0, 0, 50), VECTOR(0, 0, 0), &options);

    clock_t tic = clock();

    render(framebuffer, scene, sizeof(Object)*(num_spheres+8) / sizeof(scene[0]), &camera, &options);

    clock_t toc = clock();

    double time_taken = (double)((toc - tic) / CLOCKS_PER_SEC);
    
    printf("%d x %d (%d) pixels\n", options.width, options.height, options.width * options.height);
    printf("cast %lld rays\n", ray_count[0]);
    printf("checked %lld possible intersections\n", intersection_test_count[0]);
    printf("rendering took %f seconds\n", time_taken);
    printf("writing result to '%s'...\n", options.result);
    free(intersection_test_count);
    free(ray_count);
#ifndef VALGRIND
    write_image(0);
#endif
    return EXIT_SUCCESS;
}
