# Image Processor

A C++ command-line image processing project that applies transformations to PPM images. The implementation uses a decorator-style architecture so transformations such as rotate, flip, and sepia can be composed from command-line arguments.

## What This Demonstrates

- C++ file parsing and output generation.
- PPM image representation and transformation pipelines.
- Decorator pattern for composing image operations.
- Error handling for invalid inputs and malformed image data.

## Features

- Loads an input PPM image.
- Applies one or more transformations in order.
- Supports `rotate`, `flip`, and `sepia` transformations.
- Writes the transformed image to an output file.

## Key Files

| File | Purpose |
| --- | --- |
| `main.cc` | Command-line entry point and transformation selection. |
| `ppm.*` | PPM parsing and serialization. |
| `image.*`, `basic.*` | Base image representation. |
| `decorator.*` | Shared decorator abstraction. |
| `rotate.*`, `flip.*`, `sepia.*` | Concrete image transformations. |
| `exception.*` | Input-format error handling. |

## Build

```bash
make
```

## Usage

```bash
./a4q2 input.ppm output.ppm rotate flip sepia
```

The transformations are applied in the order provided after the output filename.
