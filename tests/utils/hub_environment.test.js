import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";

// Run the real hub module without Node globals, as in the web build. No model
// downloads or ONNX initialization are needed to test environment dispatch.
async function loadHub({ node = false, useFS = node, settings = {}, status = 200 } = {}) {
  const requests = [];
  const env = {
    useFS,
    allowLocalModels: false,
    allowRemoteModels: true,
    useBrowserCache: false,
    useFSCache: false,
    useCustomCache: false,
    localModelPath: "/models/",
    remoteHost: "https://models.example/",
    remotePathTemplate: "{model}/resolve/{revision}/",
    ...settings,
  };
  const context = vm.createContext({
    console,
    URL,
    Headers,
    Response,
    ReadableStream,
    TextDecoder,
    fetch: async (url) => {
      requests.push(String(url));
      return new Response('{"model_type":"test"}', { status });
    },
    ...(node ? { process: { platform: process.platform, release: { name: "node" }, env: {} } } : {}),
  });
  const synthetic = (exports) => new vm.SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
  }, { context });
  const core = new vm.SourceTextModule(
    await fs.promises.readFile(new URL("../../src/utils/core.js", import.meta.url), "utf8"),
    { context },
  );
  const hub = new vm.SourceTextModule(
    await fs.promises.readFile(new URL("../../src/utils/hub.js", import.meta.url), "utf8"),
    { context },
  );
  const modules = {
    fs: synthetic({ default: node ? fs : {} }),
    path: synthetic({ default: node ? path : {} }),
    "../env.js": synthetic({ apis: { IS_NODE_ENV: node }, env }),
    "./core.js": core,
  };
  await hub.link((specifier) => {
    assert.ok(specifier in modules, `Unexpected dependency: ${specifier}`);
    return modules[specifier];
  });
  await hub.evaluate();
  return { hub: hub.namespace, requests };
}

describe("Hub environment dispatch", () => {
  it("loads a model ID in a browser without process", async () => {
    const { hub, requests } = await loadHub();
    const config = await hub.getModelJSON("owner/model", "config.json");
    assert.equal(config.model_type, "test");
    assert.deepEqual(requests, ["https://models.example/owner/model/resolve/main/config.json"]);
  });

  it("keeps browser root-relative paths on the fetch path", async () => {
    const { hub, requests } = await loadHub({ settings: { allowLocalModels: true } });
    const config = await hub.getModelJSON("/models/local", "config.json");
    assert.equal(config.model_type, "test");
    assert.deepEqual(requests, ["/models/local/config.json"]);
  });

  it("supports browser progress callbacks", async () => {
    const events = [];
    const { hub } = await loadHub();
    const config = await hub.getModelJSON("owner/model", "config.json", true, {
      progress_callback: (event) => events.push(event.status),
    });
    assert.equal(config.model_type, "test");
    assert.deepEqual(events, ["initiate", "download", "progress", "done"]);
  });

  it("uses a browser custom cache without touching the network", async () => {
    const { hub, requests } = await loadHub({ settings: {
      useCustomCache: true,
      customCache: {
        match: async () => new Response('{"model_type":"cached"}'),
        put: async () => assert.fail("Unexpected cache write"),
      },
    } });
    assert.equal((await hub.getModelJSON("owner/model", "config.json")).model_type, "cached");
    assert.equal(requests.length, 0);
  });

  it("retains optional-file behavior in browsers", async () => {
    const { hub } = await loadHub({ status: 404 });
    assert.equal(await hub.getModelFile("owner/model", "optional.json", false), null);
  });

  it("retains local-only errors in browsers", async () => {
    const { hub, requests } = await loadHub();
    await assert.rejects(hub.getModelFile("owner/model", "config.json", true, {
      local_files_only: true,
    }), /local models are disabled/);
    assert.equal(requests.length, 0);
  });

  it("returns an existing absolute filesystem path in Node.js", async () => {
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), "hub-env-"));
    try {
      const filename = "model.onnx";
      await fs.promises.writeFile(path.join(directory, filename), "synthetic model placeholder");
      const { hub, requests } = await loadHub({ node: true });
      assert.equal(await hub.getModelFile(directory, filename, true, {}, true), path.join(directory, filename));
      assert.equal(requests.length, 0);
    } finally {
      await fs.promises.rm(directory, { recursive: true });
    }
  });

  it("respects disabled filesystem access in Node.js", async () => {
    const { hub, requests } = await loadHub({ node: true, useFS: false, settings: { allowLocalModels: true } });
    assert.equal((await hub.getModelJSON("/models/local", "config.json")).model_type, "test");
    assert.deepEqual(requests, ["/models/local/config.json"]);
  });
});
