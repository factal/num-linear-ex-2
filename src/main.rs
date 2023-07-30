use nalgebra::{DMatrix, DVector, ComplexField};

fn main() {
    test_calc_eigen_val_by_givens();
}

fn calc_spectral_radius(a: &DMatrix<f64>) -> f64 {
    a.complex_eigenvalues().iter().map(|x| x.abs()).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
}

fn tridiagonalize_by_householder(mat: &DMatrix<f64>) -> DMatrix<f64> {
    let dim = mat.nrows();
    let mut mat = mat.clone();

    for i in 0..dim-1 {
        let ith_col = mat.column(i);
        let a = ith_col.rows_range(i+1..dim);

        // create e_1
        let mut e1 = DVector::from_element(dim - i-1, 0.0);
        e1[0] = 1.0;

        let v = a + a.dot(&a).sqrt() * &e1;
        // correspond to \tilde{H}
        let h = DMatrix::identity(dim - i-1, dim - i-1) - 2.0 * &v * v.clone().transpose() / v.dot(&v);

        let mut h_full = DMatrix::identity(dim, dim);
        for j in i+1..dim {
            for k in i+1..dim {
                h_full[(j, k)] = h[(j-i-1, k-i-1)];
            }
        }

        mat = &h_full.transpose() * &mat * &h_full;
    }

    return mat
}
 
fn calc_eigen_val_by_givens(tri_mat: &DMatrix<f64>, max_iter: usize) -> Vec<f64> {
    let dim = tri_mat.nrows();

    let sum = tri_mat.iter().map(|x| x.abs()).sum::<f64>();
    
    let mut ans: Vec<f64> = vec![];

    for k in 1..dim+1 {
        let mut l = -sum;
        let mut r = sum;

        let mut  m = (l + r) / 2.0;

        for _ in 0..max_iter {
            m = (l + r) / 2.0;
    
            // compute p_i(m)
            let mut p = vec![1., tri_mat[0] - m];
            
            for i in 2..dim+1 {
                let new = (tri_mat[(i - 1, i - 1)] - m) * p[i-1] - tri_mat[(i-2, i-1)].powi(2) * p[i-2];
                p.push(new);
            }
    
            // compute N(n, m)
            let mut n: Vec<f64> = vec![];
            if m <= tri_mat[0] {
                n.push(0.);
            } else {
                n.push(1.);
            }
    
            for i in 1..dim {
                if p[i+1].signum() == p[i].signum() {
                    n.push(n[i-1]);
                } else {
                    n.push(n[i-1] + 1.);
                }
            }
    
            if n[dim-1] >= k as f64 {
                r = m;
            } else {
                l = m;
            }
        }

        ans.push(m);
    }

    ans
}

fn calc_eigen_val(sym_mat: &DMatrix<f64>, max_iter: usize) -> Vec<f64> {
    let tri_mat = tridiagonalize_by_householder(sym_mat);
    let eigen_vals = calc_eigen_val_by_givens(&tri_mat, max_iter);

    eigen_vals
}

fn check_tridiagonal(mat: &DMatrix<f64>, error: f64) -> bool {
    let dim = mat.nrows();

    for i in 0..dim {
        for j in 0..dim {
            if i != j && (i != j + 1 && i + 1 != j) {
                if mat[(i, j)].abs() > error {
                    return false;
                }
            }

            if i == j + 1 || i + 1 == j {
                if (mat[(i, j)] - mat[(j, i)]).abs() > error {
                    return false;
                }
            }
        }
    }

    return true;
}

fn create_random_tridiag(size: usize) -> DMatrix<f64> {
    let mut mat = DMatrix::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            if i == j {
                mat[(i, j)] = rand::random::<f64>();
            } else if i + 1 == j {
                mat[(i, j)] = rand::random::<f64>();
            } else if i == j + 1 {
                mat[(i, j)] = mat[(j, i)]
            }
        }
    }

    return mat;
}

fn test_calc_eigen_val_by_givens() {
    const MAX_ITER: usize = 1_000;

    let mut test_data: Vec<DMatrix<f64>> = vec![];

    for _ in 0..MAX_ITER {
        test_data.push(create_random_tridiag(5));
    }

    let mut target_eigen_vals: Vec<Vec<f64>>  = vec![];
    let mut test_eigen_vals: Vec<Vec<f64>> = vec![];

    for i in 0..MAX_ITER {
        let mut target = test_data[i].complex_eigenvalues().iter().map(|x| x.re).collect::<Vec<f64>>();
        target.sort_by(|a, b| a.partial_cmp(b).unwrap());
        target_eigen_vals.push(target);
        let mut test = calc_eigen_val_by_givens(&test_data[i], 10_000);
        test.sort_by(|a, b| a.partial_cmp(b).unwrap());
        test_eigen_vals.push(test);
    }

    let mut errors: Vec<f64> = vec![];

    for i in 0..MAX_ITER {
        let mut error = 0.0;
        for j in 0..5 {
            error += (target_eigen_vals[i][j] - test_eigen_vals[i][j]).powi(2);
        }
        errors.push(error.sqrt());
    }

    let (max_index, max) = errors.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap();

    println!("max error: {}", max);
    println!("the matrix: {:}", test_data[max_index]);
}

fn test_ex_10_7 () {
    let test = DMatrix::from_row_slice(5, 5,
        &[2., 1., 2., 2., 2.,
          1., 2., 1., 2., 2.,
          2., 1., 2., 1., 2.,
          2., 2., 1., 2., 1.,
          2., 2., 2., 1., 2.,]);
    
    let ex10_9_result = tridiagonalize_by_householder(&test);
    
    println!("tridiagonalized: {:6}", ex10_9_result);
    println!("spectra: ρ(T) = {:6}, ρ(A) = {:6}", calc_spectral_radius(&ex10_9_result), calc_spectral_radius(&test));
    println!("eigen value: {:?}", calc_eigen_val_by_givens(&ex10_9_result, 10_000));
    println!("is triagonal: {}", check_tridiagonal(&ex10_9_result, 1e-10));
}