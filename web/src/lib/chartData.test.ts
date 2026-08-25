import { describe, expect, test } from "bun:test";

import type { RunDetail } from "../api";
import { alignMetric } from "./chartData";

const run = (id: string, points: Array<[number, number]>): RunDetail =>
  ({
    id,
    config: {},
    environment: {},
    metrics: points.map(([step, loss]) => ({ type: "train", step, epoch: 0, loss })),
  }) as unknown as RunDetail;

describe("alignMetric", () => {
  test("aligns sparse runs on the union of steps", () => {
    expect(
      alignMetric(
        [
          run("a", [
            [1, 3],
            [3, 1],
          ]),
          run("b", [[2, 2]]),
        ],
        "loss",
      ),
    ).toEqual({
      x: [1, 2, 3],
      values: [
        [3, null, 1],
        [null, 2, null],
      ],
    });
  });
});
