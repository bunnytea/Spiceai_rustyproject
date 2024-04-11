use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufWriter, Result, Write};

fn main() {
    let mut xr = np_exp(&np_sin(&np_power(&np_linspace(0.0, 9.0, 256), 2.0)));
    let mut xi = np_zeros(256);
    let mut fr = xr.clone();
    let mut fi = xi.clone();
    fft(&mut fr, &mut fi);
    vector2csv(&fr, "reals.csv").expect("Failed to write to CSV");
    vector2csv(&fi, "imags.csv").expect("Failed to write to CSV");

  let mut signalminimum = np_min(&xr);
  let mut shiftedsignal=np_add(&xr,&scale_vector(&np_ones(xr.len()),(-1.0)*signalminimum));
  let mut maximumthing = np_max(&shiftedsignal);
  let mut normalizedsignal=scale_vector(&shiftedsignal,(100.0/maximumthing));
  let mut timedomainplotdata = np_zeroes(120, 256);
  let mut intsignal = float2int(&normalizedsignal);
  println!("{:?}", intsignal);
  for (k, &value) in intsignal.iter().enumerate() {
      if value <= 100 {
          timedomainplotdata[k][value as usize] = 150.0;
      }
  }
  //print_matrix(&timedomainplotdata);
  //flipud(&mut timedomainplotdata);
  //fliplr(&mut timedomainplotdata);
  array2csv(&timedomainplotdata, "timedomainplotdata.csv").expect("Failed to write to CSV");
  let timedomainimage = float_array2int(&timedomainplotdata);
  save_bitmap_image(&timedomainimage, "output.bmp");
  let mut magnitudespectrum=np_sqrt(&np_add(&np_power(&fr,2.0), &np_power(&fi,2.0)));
  println!("{:?}", magnitudespectrum);


  let mut signalminimum = np_min(&magnitudespectrum);
  let mut shiftedsignal=np_add(&magnitudespectrum,&scale_vector(&np_ones(xr.len()),(-1.0)*signalminimum));
  let mut maximumthing = np_max(&shiftedsignal);
  let mut normalizedsignal=scale_vector(&shiftedsignal,(100.0/maximumthing));
  let mut freqdomainplotdata = np_zeroes(120, 256);
  let mut intsignal = float2int(&normalizedsignal);
  println!("{:?}", intsignal);
  for (k, &value) in intsignal.iter().enumerate() {
      if value <= 100 {
          freqdomainplotdata[k][value as usize] = 150.0;
      }
  }
  array2csv(&freqdomainplotdata,  "freqdomainplotdata.csv").expect("Failed to write to CSV");



}

// Converts a vector of vectors to CSV
fn array2csv(data: &Vec<Vec<f64>>, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    let mut w = BufWriter::new(file);
    for row in data {
        let row_string = row
            .iter()
            .map(|&value| value.to_string())
            .collect::<Vec<String>>()
            .join(",");
        w.write_all(row_string.as_bytes())?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

// Converts a vector to CSV
fn vector2csv(data: &Vec<f64>, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    let mut w = BufWriter::new(file);
    let data_string = data
        .iter()
        .map(|&value| value.to_string())
        .collect::<Vec<String>>()
        .join(",");
    w.write_all(data_string.as_bytes())?;
    Ok(())
}

// Function to compute complex exponential
fn complex_exp(re: f64, im: f64) -> (f64, f64) {
    (re.exp() * im.cos(), re.exp() * im.sin())
}

// Function to add two complex numbers
fn complex_add((re1, im1): (f64, f64), (re2, im2): (f64, f64)) -> (f64, f64) {
    (re1 + re2, im1 + im2)
}

// Function to multiply two complex numbers
fn complex_multiply((re1, im1): (f64, f64), (re2, im2): (f64, f64)) -> (f64, f64) {
    (re1 * re2 - im1 * im2, re1 * im2 + im1 * re2)
}

// Recursive FFT function
fn fft(xr: &mut [f64], xi: &mut [f64]) {
    let n = xr.len();
    if n == 1 {
        return;
    } else {
        let mut even_r = Vec::with_capacity(n / 2);
        let mut even_i = Vec::with_capacity(n / 2);
        let mut odd_r = Vec::with_capacity(n / 2);
        let mut odd_i = Vec::with_capacity(n / 2);

        for i in 0..n {
            if i % 2 == 0 {
                even_r.push(xr[i]);
                even_i.push(xi[i]);
            } else {
                odd_r.push(xr[i]);
                odd_i.push(xi[i]);
            }
        }

        fft(&mut even_r, &mut even_i);
        fft(&mut odd_r, &mut odd_i);

        let mut factor_r = Vec::with_capacity(n);
        let mut factor_i = Vec::with_capacity(n);
        for k in 0..n {
            let angle = -2.0 * PI * k as f64 / n as f64;
            let (re, im) = complex_exp(0.0, angle);
            factor_r.push(re);
            factor_i.push(im);
        }

        for k in 0..n / 2 {
            let (t_r, t_i) = complex_multiply((factor_r[k], factor_i[k]), (odd_r[k], odd_i[k]));
            xr[k] = even_r[k] + t_r;
            xi[k] = even_i[k] + t_i;
            xr[k + n / 2] = even_r[k] - t_r;
            xi[k + n / 2] = even_i[k] - t_i;
        }
    }
}

// emulates np.arange
fn np_arange(stop: i32) -> Vec<i32> {
    (0..stop).collect()
}

//converts vector of floats to ints
fn float2int(float_vector: &Vec<f64>) -> Vec<i32> {
    float_vector.iter().map(|&x| x as i32).collect()
}

//converts vector of ints to floats
fn int2float(int_vector: &Vec<i32>) -> Vec<f64> {
    int_vector.iter().map(|&x| x as f64).collect()
}

//multiplies vector by scalar
fn scale_vector(vector: &Vec<f64>, scalar: f64) -> Vec<f64> {
    vector.iter().map(|&x| x * scalar).collect()
}

// emulates np.dot()
fn np_dot(arr1: &[f64], arr2: &[f64]) -> f64 {
    arr1.iter().zip(arr2.iter()).map(|(&x, &y)| x * y).sum()
}

// emulates np.sin()
fn np_sin(arr: &[f64]) -> Vec<f64> {
    arr.iter().map(|&x| x.sin()).collect()
}

// emulates np.cos()
fn np_cos(arr: &[f64]) -> Vec<f64> {
    arr.iter().map(|&x| x.cos()).collect()
}
//emulates np.concatenate()
fn np_concatenate<T: Clone>(arr1: &[T], arr2: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(arr1.len() + arr2.len());
    result.extend_from_slice(arr1);
    result.extend_from_slice(arr2);
    result
}

// emulates x[:index]
fn slice_start<T: Clone>(arr: &[T], index: usize) -> Vec<T> {
    arr[..index].to_vec()
}

// emulates x[index:]
fn slice_end<T: Clone>(arr: &[T], index: usize) -> Vec<T> {
    arr[index..].to_vec()
}

// emulates np.add
fn np_add(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x + y).collect()
}

// emulates np.multiply
fn np_multiply(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x * y).collect()
}

// emulates np.divide
fn np_divide(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x / y).collect()
}

// emulates np.exp
fn np_exp(vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|&x| x.exp()).collect()
}

// emulates np.power
fn np_power(vec: &Vec<f64>, exponent: f64) -> Vec<f64> {
    vec.iter().map(|&x| x.powf(exponent)).collect()
}

// emulates np.sqrt
fn np_sqrt(vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|&x| x.sqrt()).collect()
}

// emulates np.ones
fn np_ones(size: usize) -> Vec<f64> {
    vec![1.0; size]
}

// emulates np.zeros
fn np_zeros(size: usize) -> Vec<f64> {
    vec![0.0; size]
}
// emulates np.linspace
fn np_linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / ((num - 1) as f64);
    (0..num).map(|i| start + step * (i as f64)).collect()
}

// Function to save image data into a bitmap file
fn save_bitmap_image(image_data: &[Vec<u8>], filename: &str) {
    // Extract width and height from image_data
    let width = image_data[0].len();
    let height = image_data.len();

    // Calculate image size in bytes
    let image_size = width * height * 3;

    // Create a new file to write the bitmap image
    let mut file = File::create(filename).expect("Unable to create file");

    // Write BMP header
    let header: Vec<u8> = vec![
        0x42,
        0x4D, // BM
        (54 + image_size) as u8,
        0x00,
        0x00,
        0x00, // File size in bytes
        0x00,
        0x00,
        0x00,
        0x00, // Reserved
        0x36,
        0x00,
        0x00,
        0x00, // Image data offset
        0x28,
        0x00,
        0x00,
        0x00, // Header size
        width as u8,
        0x00,
        0x00,
        0x00, // Image width
        height as u8,
        0x00,
        0x00,
        0x00, // Image height
        0x01,
        0x00, // Number of color planes
        0x18,
        0x00, // Bits per pixel (24-bit)
        0x00,
        0x00,
        0x00,
        0x00, // Compression method (none)
        (image_size) as u8,
        0x00,
        0x00,
        0x00, // Image size (unspecified)
        0x00,
        0x00,
        0x00,
        0x00, // Horizontal resolution (unspecified)
        0x00,
        0x00,
        0x00,
        0x00, // Vertical resolution (unspecified)
        0x00,
        0x00,
        0x00,
        0x00, // Number of colors in palette (default)
        0x00,
        0x00,
        0x00,
        0x00, // Number of important colors (all)
    ];

    file.write_all(&header)
        .expect("Unable to write header to file");

    // Write image data
    for row in image_data {
        for pixel in row {
            // For grayscale, use the same intensity for all channels
            file.write_all(&[*pixel, *pixel, *pixel])
                .expect("Unable to write pixel to file");
        }
    }

    println!("Bitmap image successfully created!");
}

//emulates np.max
fn np_max(vec: &[f64]) -> f64 {
    let mut max = vec[0];
    for &element in vec.iter() {
        if element > max {
            max = element;
        }
    }
    max
}

//emulates np.min
fn np_min(vec: &[f64]) -> f64 {
    let mut min = vec[0];
    for &element in vec.iter() {
        if element < min {
            min = element;
        }
    }
    min
}

//emulates np.min for integers
fn mint(vec: &[i32]) -> i32 {
    let mut min = vec[0];
    for &element in vec.iter() {
        if element < min {
            min = element;
        }
    }
    min
}
//emulates np.max for integers
fn maxint(vec: &[i32]) -> i32 {
    let mut max = vec[0];
    for &element in vec.iter() {
        if element > max {
            max = element;
        }
    }
    max
}



// emulates np.zeroes
fn np_zeroes(num_columns: usize, num_rows: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; num_columns]; num_rows]
}

fn flipud(matrix: &mut Vec<Vec<f64>>) {
    matrix.reverse();
}

fn fliplr(matrix: &mut Vec<Vec<f64>>) {
    for row in matrix.iter_mut() {
        row.reverse();
    }
}

fn print_matrix(matrix: &[Vec<f64>]) {
  for row in matrix {
    println!("{:?}", row);
  }
}

fn floatarray2int(float_matrix: &[Vec<f64>]) -> Vec<Vec<i32>> {
    float_matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| x.round() as i32)
                .collect::<Vec<i32>>()
        })
        .collect()
}

fn float_array2int(matrix: &[Vec<f64>]) -> Vec<Vec<u8>> {
    matrix.iter().map(|row| {
        row.iter().map(|&x| x.round() as u8).collect()
    }).collect()
}
