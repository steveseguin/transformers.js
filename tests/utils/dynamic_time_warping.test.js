import assert from "node:assert/strict";
import { dynamic_time_warping } from "../../src/utils/maths.js";

function pathCost(matrix, [rows, cols]) {
  assert.equal(rows.length, cols.length);
  assert.deepEqual([rows[0], cols[0]], [0, 0]);
  assert.deepEqual([rows.at(-1), cols.at(-1)], [matrix.length - 1, matrix[0].length - 1]);
  let cost = 0;
  for (let k = 0; k < rows.length; ++k) {
    assert.ok(rows[k] >= 0 && rows[k] < matrix.length);
    assert.ok(cols[k] >= 0 && cols[k] < matrix[0].length);
    if (k) {
      const step = [rows[k] - rows[k - 1], cols[k] - cols[k - 1]];
      assert.ok((step[0] === 1 && step[1] === 1) || (step[0] === 1 && step[1] === 0) || (step[0] === 0 && step[1] === 1));
    }
    cost += matrix[rows[k]][cols[k]];
  }
  return cost;
}

// Enumerate paths instead of reproducing the production dynamic-programming recurrence.
function minimumPathCost(matrix, row = 0, col = 0) {
  if (row === matrix.length - 1 && col === matrix[0].length - 1) {
    return matrix[row][col];
  }
  const tails = [];
  if (row + 1 < matrix.length) tails.push(minimumPathCost(matrix, row + 1, col));
  if (col + 1 < matrix[0].length) tails.push(minimumPathCost(matrix, row, col + 1));
  if (row + 1 < matrix.length && col + 1 < matrix[0].length) tails.push(minimumPathCost(matrix, row + 1, col + 1));
  return matrix[row][col] + Math.min(...tails);
}

describe("dynamic time warping", () => {
  it("selects a minimum-cost predecessor when diagonal and vertical tie", () => {
    const matrix = [[0, 0], [1, 0]];
    const path = dynamic_time_warping(matrix);
    assert.equal(pathCost(matrix, path), 0);
    assert.deepEqual(path, [[0, 0, 1], [0, 1, 1]]);
  });

  it("preserves horizontal preference when all predecessors tie", () => {
    const matrix = [[0, 0], [0, 0]];
    assert.deepEqual(dynamic_time_warping(matrix), [[0, 1, 1], [0, 0, 1]]);
  });

  it("preserves horizontal preference when horizontal and diagonal tie", () => {
    const matrix = [[0, 1], [0, 0]];
    assert.deepEqual(dynamic_time_warping(matrix), [[0, 1, 1], [0, 0, 1]]);
  });

  it("handles single-cell, single-row and single-column matrices", () => {
    for (const matrix of [[[3]], [[1, -2, 4]], [[1], [-2], [4]]]) {
      assert.equal(pathCost(matrix, dynamic_time_warping(matrix)), minimumPathCost(matrix));
    }
  });

  it("preserves horizontal preference when horizontal and vertical tie", () => {
    const matrix = [[0, -1], [-1, 0]];
    assert.deepEqual(dynamic_time_warping(matrix), [[0, 1, 1], [0, 0, 1]]);
  });

  it("preserves strict diagonal, vertical and horizontal minima", () => {
    for (const matrix of [[[0, 2], [3, 0]], [[0, -2], [3, 0]], [[0, 2], [-3, 0]]]) {
      assert.equal(pathCost(matrix, dynamic_time_warping(matrix)), minimumPathCost(matrix));
    }
  });

  it("keeps token boundaries optimal for repeated smoothed attention values", () => {
    // Float32 fixture after column normalization and width-7 median filtering.
    const matrix = [
      [0.17570410668849945, 0.17570410668849945, 0.17570410668849945, 0.17570410668849945, 0.9514170289039612, 1.0704151391983032, 1.0704151391983032, 1.0704151391983032],
      [0.43043941259384155, 0.43043941259384155, 0.43043941259384155, 0.43043941259384155, 0.2652045786380768, 0.2652045786380768, 0.2652045786380768, 0.2652045786380768],
      [0.016764281317591667, -0.4973004162311554, -0.4973004162311554, -0.4973004162311554, -0.4973004162311554, -0.4973004162311554, -0.4973004162311554, -0.4973004162311554]
    ];
    const path = dynamic_time_warping(matrix);
    assert.ok(Math.abs(pathCost(matrix, path) - minimumPathCost(matrix)) < 1e-12);
    const [rows, cols] = path;
    const jumpFrames = cols.filter((_, k) => k === 0 || rows[k] !== rows[k - 1]);
    assert.deepEqual(jumpFrames, [0, 1, 1]);
  });

  it("matches every path for small integer matrices including repeated values", () => {
    for (const [height, width] of [[2, 2], [2, 3], [3, 2], [3, 3]]) {
      const cells = height * width;
      for (let pattern = 0; pattern < 3 ** cells; ++pattern) {
        let value = pattern;
        const matrix = Array.from({ length: height }, () => Array.from({ length: width }, () => {
          const entry = value % 3 - 1;
          value = Math.floor(value / 3);
          return entry;
        }));
        assert.equal(pathCost(matrix, dynamic_time_warping(matrix)), minimumPathCost(matrix), JSON.stringify(matrix));
      }
    }
  });
});
