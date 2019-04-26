#!/usr/bin/env python3
# Converts STL/OBJ files into a JSON file
# formatted for use with mrraytracer.
# Allows a user to specify scale, rotate
# and translation transformations on the
# input file.
import argparse
import json
import sys
import math

import numpy
from stl import mesh
from enum import Enum
from multiprocessing import Pool

output = {
    "command_line": "",
    "camera_eye": (0.0, 100.0, 20.0),
    "camera_up": (0, 0, -1),
    "camera_view": (0, -1, 0),
    "x_resolution": 200,
    "y_resolution": 200,
    "viewport_left": -1,
    "viewport_top": 1,
    "viewport_right": 1,
    "viewport_bottom": -1,
    "background": (0.29, 0.81, 0.91),
    "perspective_focal_length": 1.0,
    "blinn_phong_shader": {
        "ambient_coefficient": 0.05,
        "diffuse_coefficient": 0.5,
        "specular_coefficient": 0.25,
        "ambient_color": (1.0, 1.0, 1.0)
    },
    "lights": [
        {
            "location": (-300.0, 200.0, 0.0),
            "intensity": 1.1,
            "color": (1.0, 1.0, 1.0)
        },
        {
            "location": (-100.0, 0.0, 0.0),
            "intensity": 0.25,
            "color": (1.0, 1.0, 1.0)
        },
        {
            "location": (0.0, 250.0, 50.0),
            "intensity": 0.75,
            "color": (1.0, 1.0, 1.0)
        }
    ],
    "triangles": []
};

def vector_to_matrix(v):
    return numpy.matrix([
        [v[0]],
        [v[1]],
        [v[2]],
        [1]
    ])

# 3D Transformations - Slide 3
def scale(vector, x, y, z):
    scale_matrix = numpy.matrix([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ])

    return scale_matrix * vector_to_matrix(vector)

# 3D Transformations - Slide 2
def translate(vector, x, y, z):
    translate_matrix = numpy.matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    return translate_matrix * vector_to_matrix(vector)

# 3D Transformations - Slide 4
def rotate(vector, axis, degrees):
    assert(axis == 'x' or axis == 'y' or axis == 'z')

    theta = math.radians(degrees)
    rotate_matrix = numpy.identity(4)
    if axis == 'x':
        rotate_matrix[1][1] = math.cos(theta)
        rotate_matrix[1][2] = -math.sin(theta)
        rotate_matrix[2][1] = math.sin(theta)
        rotate_matrix[2][2] = math.cos(theta)
    elif axis == 'y':
        rotate_matrix[0][0] = math.cos(theta)
        rotate_matrix[0][1] = math.sin(theta)
        rotate_matrix[2][0] = -math.sin(theta)
        rotate_matrix[2][2] = math.cos(theta)
    else:
        rotate_matrix[0][0] = math.cos(theta)
        rotate_matrix[0][1] = -math.sin(theta)
        rotate_matrix[1][0] = math.sin(theta)
        rotate_matrix[1][1] = math.cos(theta)

    return rotate_matrix * vector_to_matrix(vector)


scale_factors = [1.0, 1.0, 1.0]
rotate_factors = [0, 0, 0]
translate_factors = [1.0, 1.0, 1.0]
def perform_translations(vector):
    global scale_factors
    global rotate_factors
    global translate_factors

    for v in vector:
        # Scale -> Rotate -> Translate
        if scale_factors != [1.0, 1.0, 1.0]:        
            scaled = scale(v, scale_factors[0], scale_factors[1], scale_factors[2])                    
            v[0] = scaled[0]
            v[1] = scaled[1]
            v[2] = scaled[2]

        if rotate_factors != [0, 0, 0]:
            rotated = rotate(v, 'x', rotate_factors[0])
            rotated = rotate(rotated, 'y', rotate_factors[1])
            rotated = rotate(rotated, 'z', rotate_factors[2])
            
            v[0] = rotated[0]
            v[1] = rotated[1]
            v[2] = rotated[2]

        if translate_factors != [1.0, 1.0, 1.0]:
            translated = translate(v, translate_factors[0], translate_factors[1], translate_factors[2])
            v[0] = translated[0]
            v[1] = translated[1]
            v[2] = translated[2]
    
    return vector

def main():
    global scale_factors
    global rotate_factors
    global translate_factors

    parser = argparse.ArgumentParser(description="Converts models into a mrraytracer JSON file.")
    
    parser.add_argument('inputFile', type=argparse.FileType('r'), default=sys.stdin, help="STL/OBJ model")
    parser.add_argument('--scale', type=str, help="scale input by x,y,z where 1 = 100%%")
    parser.add_argument('--translate', type=str, help="translate input by x,y,z")
    parser.add_argument('--rotate', type=str, help="rotate input by x,y,z degrees about x,y,z axes")

    parser.add_argument('outputFile', type=argparse.FileType('w'), default=sys.stdout, help="mrraytracer .json output")
    args = parser.parse_args(sys.argv[1:])

    # Store the command we used to generate the JSON file 
    # inside of the JSON file in case anyone is curious.
    output["command_line"] = " ".join(sys.argv)

    # 3D Transformations - Slide 3
    if args.scale and len(args.scale.split(',')) == 3:
        user_scale = numpy.array([float(v) for v in args.scale.split(',')])
        scale_factors = (user_scale * scale_factors).tolist()
    else:
        user_scale = numpy.ones(3) * 1
        scale_factors = (user_scale * scale_factors).tolist()

    # 3D Transformations - Slide 3
    if args.rotate and len(args.rotate.split(',')) == 3:
        rotate_factors = [float(v) for v in args.rotate.split(',')]

    # 3D Transformations - Slide 2
    if args.translate and len(args.translate.split(',')) == 3:
        user_translate = numpy.array([float(v) for v in args.translate.split(',')])
        translate_factors = (user_translate * translate_factors).tolist()
    else:
        user_translate = numpy.ones(3) * 1
        translate_factors = (user_translate * translate_factors).tolist()
    
    vectors = None

    fileName = args.inputFile.name
    if ".obj" in fileName:
        vertices = []
        faces = []

        lines = args.inputFile.readlines()
        for line in lines:
            tokens = line.split()
            if line.startswith('v') and not (line.startswith('vt') or line.startswith('vn') or line.startswith('vp')):
                assert(len(tokens) == 4)
                vertices.append([ float(tok.split('/')[0]) for tok in tokens[1:] ])
            elif line.startswith('f'):
                assert(len(tokens) == 4)
                faces.append([ int(tok.split('/')[0]) for tok in tokens[1:] ])
            else:
                # comment
                pass

        vectors = []
        for face in faces:
            a = [v for v in vertices[face[0] - 1]]
            b = [v for v in vertices[face[1] - 1]]
            c = [v for v in vertices[face[2] - 1]]
            vectors.append(numpy.array([a, b, c]))
    elif ".stl" in fileName:
        # Create mesh object from input file
        input_mesh = mesh.Mesh.from_file(args.inputFile.name)
        vectors = input_mesh.vectors
    else:
        print("Unsupported file type.")
        exit(1)

    # Run transformation across multiple threads
    transform_pool = Pool(processes=8)
    print(f"Transforming {len(vectors)} vectors")
    triangles = transform_pool.map(perform_translations, vectors)
    output["triangles"].append([{
        "color": (0.8, 0.95, 0.7),
        "shininess": 4.0,
        "a": [v for v in vector[0].tolist()],
        "b": [v for v in vector[1].tolist()],
        "c": [v for v in vector[2].tolist()]
    } for vector in triangles])

    output["triangles"] = output["triangles"][0]

    args.outputFile.write(json.dumps(output))    

if __name__ == '__main__':
    main()