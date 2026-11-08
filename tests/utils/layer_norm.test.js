import { Tensor, layer_norm } from "../../src/transformers.js";
import { init } from "../init.js";

init();

describe("layer_norm epsilon", () => {
  it.each([
    ["default epsilon with low variance", [-0.0001, 0.0001], undefined, [-0.0316069770620507, 0.0316069770620507]],
    ["custom epsilon", [-1, 1], { eps: 1 }, [-Math.SQRT1_2, Math.SQRT1_2]],
    ["nonzero mean", [2, 4], { eps: 3 }, [-0.5, 0.5]],
    ["zero epsilon", [-2, 2], { eps: 0 }, [-1, 1]],
    ["constant row", [7, 7], undefined, [0, 0]],
    ["single feature", [3], undefined, [0]],
  ])("uses variance plus epsilon for %s", (_, data, options, expected) => {
    const input = new Tensor("float32", data, [1, data.length]);
    const originalData = Array.from(input.data);
    const output = layer_norm(input, [data.length], options);

    expect(output.dims).toEqual(input.dims);
    expect(output.type).toBe("float32");
    expect(Array.from(input.data)).toEqual(originalData);
    expect(output.data).not.toBe(input.data);
    for (let i = 0; i < expected.length; ++i) {
      expect(output.data[i]).toBeCloseTo(expected[i], 6);
    }
  });

  it("normalizes each batch row with its own variance", () => {
    const input = new Tensor("float32", [-1, 1, 8, 12, 5, 5], [3, 2]);
    const output = layer_norm(input, [2], { eps: 3 });
    const expected = [-0.5, 0.5, -2 / Math.sqrt(7), 2 / Math.sqrt(7), 0, 0];

    expect(output.dims).toEqual([3, 2]);
    for (let i = 0; i < expected.length; ++i) {
      expect(output.data[i]).toBeCloseTo(expected[i], 6);
    }
  });
});
