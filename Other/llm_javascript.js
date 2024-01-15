console.log("Hello World!")

// Define matrix class 
class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = [];
            
        // Initialize matrix
        for (let i = 0; i < this.rows; i++) {
            let arr = [];
            for (let j = 0; j < this.cols; j++) {
                arr.push(0);
            }
            this.data.push(arr);
        }
    }

    // Construct matrix from array of array
    fromArray(arr) {
        return new Matrix(arr.length, 1).map((e, i) => arr[i]);
    }

    // multiple matrix
    mul(other) {
        if (this.cols !== other.rows) {
            console.log("Columns of A must match rows of    B.");
            return undefined;
        }
        let result = new Matrix(this.rows, other.cols);
        // Dot product
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                // Sum of pairwise products
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        } 
    }
}

// Define matrix class
const A = new Matrix(2, 3);
const B = new Matrix(3, 2);

A.mul(B);