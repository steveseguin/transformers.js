import { AutoTokenizer, WhisperTokenizer } from "../src/tokenizers.js";
import * as TOKENIZER_TESTS from "./models/all_tokenization_tests.js";

import { compare } from "./test_utils.js";

const MAX_LOAD_TIME = 10_000;
const MAX_EXECUTION_TIME = 10_000;

describe("Tokenizers (model-specific)", () => {
  for (const [tokenizer_name, { TOKENIZER_CLASS, TEST_CONFIG }] of Object.entries(TOKENIZER_TESTS)) {
    describe(tokenizer_name, () => {
      for (const model_id in TEST_CONFIG) {
        describe(model_id, () => {
          let tokenizer;
          beforeAll(async () => {
            tokenizer = await TOKENIZER_CLASS.from_pretrained(model_id);
          }, MAX_LOAD_TIME);

          for (const [test_name, test_case] of Object.entries(TEST_CONFIG[model_id])) {
            test(test_name, () => {
              const ids = tokenizer.encode(test_case.text);
              expect(ids).toEqual(test_case.ids);
              const tokens = tokenizer.tokenize(test_case.text);
              expect(tokens).toEqual(test_case.tokens);
              const decoded = tokenizer.decode(test_case.ids);
              expect(decoded).toEqual(test_case.decoded);
            });
          }
        });
      }
    });
  }
});

// Tests to ensure that no matter what, the correct tokenization is returned.
// This is necessary since there are sometimes bugs in the transformers library.
describe("Tokenizers (hard-coded)", () => {
  const TESTS = {
    "Xenova/llama-tokenizer": [
      // Test legacy compatibility
      {
        // legacy unset => legacy=true
        // NOTE: While incorrect, it is necessary to match legacy behaviour
        data: {
          "<s>\n": [1, 29871, 13],
        },
        legacy: null,
      },
      {
        // override legacy=true (same results as above)
        data: {
          "<s>\n": [1, 29871, 13],
        },
        legacy: true,
      },
      {
        // override legacy=false (fixed results)
        data: {
          "<s>\n": [1, 13],
        },
        legacy: false,
      },
    ],

    "Xenova/llama-tokenizer_new": [
      // legacy=false
      {
        data: {
          " </s> 1  2   3    4   ": [259, 2, 29871, 29896, 259, 29906, 1678, 29941, 268, 29946, 1678],
          "<s>\n": [1, 13],
          "</s>test</s>": [2, 1688, 2],
          " </s> test </s> ": [259, 2, 1243, 29871, 2, 29871],
          "A\n'll": [319, 13, 29915, 645],
          "Hey </s>. how are you": [18637, 29871, 2, 29889, 920, 526, 366],
          "  Hi  Hello  ": [259, 6324, 29871, 15043, 259],
        },
        reversible: true,
        legacy: null,
      },
      {
        // override legacy=true (incorrect results, but necessary to match legacy behaviour)
        data: {
          "<s>\n": [1, 29871, 13],
        },
        legacy: true,
      },
    ],

    // legacy=false
    "Xenova/t5-tokenizer-new": [
      {
        data: {
          // https://github.com/huggingface/transformers/pull/26678
          // ['▁Hey', '▁', '</s>', '.', '▁how', '▁are', '▁you']
          "Hey </s>. how are you": [9459, 3, 1, 5, 149, 33, 25],
        },
        reversible: true,
        legacy: null,
      },
      {
        data: {
          "</s>\n": [1, 3],
          "A\n'll": [71, 3, 31, 195],
        },
        reversible: false,
        legacy: null,
      },
    ],
  };

  // Re-use the same tests for the llama2 tokenizer
  TESTS["Xenova/llama2-tokenizer"] = TESTS["Xenova/llama-tokenizer_new"];

  for (const [tokenizerName, test_data] of Object.entries(TESTS)) {
    it(
      tokenizerName,
      async () => {
        for (const { data, reversible, legacy } of test_data) {
          const tokenizer = await AutoTokenizer.from_pretrained(tokenizerName, { legacy });

          for (const [text, expected] of Object.entries(data)) {
            const token_ids = tokenizer.encode(text, { add_special_tokens: false });
            expect(token_ids).toEqual(expected);

            // If reversible, test that decoding produces the original text
            if (reversible) {
              const decoded = tokenizer.decode(token_ids);
              expect(decoded).toEqual(text);
            }
          }
        }
      },
      MAX_EXECUTION_TIME,
    );
  }
});

describe("Tokenizer padding/truncation", () => {
  const inputs = ["a", "b c"];
  const text_pair = ["d e", "f g h"];

  it("should create a jagged array", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

    {
      // support jagged array if `return_tensor=false`
      const output = tokenizer(inputs, {
        return_tensor: false,
      });
      const expected = {
        input_ids: [
          [101, 1037, 102],
          [101, 1038, 1039, 102],
        ],
        attention_mask: [
          [1, 1, 1],
          [1, 1, 1, 1],
        ],
        token_type_ids: [
          [0, 0, 0],
          [0, 0, 0, 0],
        ],
      };
      compare(output, expected);
    }

    {
      const output = tokenizer(inputs, {
        return_tensor: false,
        truncation: true,
        add_special_tokens: false,
      });
      const expected = {
        input_ids: [[1037], [1038, 1039]],
        attention_mask: [[1], [1, 1]],
        token_type_ids: [[0], [0, 0]],
      };
      compare(output, expected);
    }
  });

  it(
    "should create a tensor",
    async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

      {
        // Expected to throw error if jagged array
        expect(() => tokenizer(inputs)).toThrowError("Unable to create tensor");
      }

      {
        // Truncation
        const { input_ids, attention_mask, token_type_ids } = tokenizer(inputs, {
          truncation: true,
          max_length: 1,
          add_special_tokens: false,
        });

        expect(input_ids.tolist()).toEqual([[1037n], [1038n]]);
        expect(attention_mask.tolist()).toEqual([[1n], [1n]]);
        expect(token_type_ids.tolist()).toEqual([[0n], [0n]]);
      }
      {
        // Truncation w/ text pair
        // TODO
      }

      {
        // Padding
        const { input_ids, attention_mask, token_type_ids } = tokenizer(inputs, {
          padding: true,
          add_special_tokens: false,
        });

        expect(input_ids.tolist()).toEqual([
          [1037n, 0n],
          [1038n, 1039n],
        ]);
        expect(attention_mask.tolist()).toEqual([
          [1n, 0n],
          [1n, 1n],
        ]);
        expect(token_type_ids.tolist()).toEqual([
          [0n, 0n],
          [0n, 0n],
        ]);
      }
      {
        // Padding w/ text pair
        const { input_ids, attention_mask, token_type_ids } = tokenizer(inputs, {
          text_pair,
          padding: true,
          add_special_tokens: false,
        });

        expect(input_ids.tolist()).toEqual([
          [1037n, 1040n, 1041n, 0n, 0n],
          [1038n, 1039n, 1042n, 1043n, 1044n],
        ]);
        expect(attention_mask.tolist()).toEqual([
          [1n, 1n, 1n, 0n, 0n],
          [1n, 1n, 1n, 1n, 1n],
        ]);
        expect(token_type_ids.tolist()).toEqual([
          [0n, 1n, 1n, 0n, 0n],
          [0n, 0n, 1n, 1n, 1n],
        ]);
      }

      {
        // Truncation + padding
        const { input_ids, attention_mask, token_type_ids } = tokenizer(["a", "b c", "d e f"], {
          padding: true,
          truncation: true,
          add_special_tokens: false,
          max_length: 2,
        });

        expect(input_ids.tolist()).toEqual([
          [1037n, 0n],
          [1038n, 1039n],
          [1040n, 1041n],
        ]);
        expect(attention_mask.tolist()).toEqual([
          [1n, 0n],
          [1n, 1n],
          [1n, 1n],
        ]);
        expect(token_type_ids.tolist()).toEqual([
          [0n, 0n],
          [0n, 0n],
          [0n, 0n],
        ]);
      }
    },
    MAX_EXECUTION_TIME,
  );
});

describe("Token type ids", () => {
  it(
    "should correctly add token type ids",
    async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

      const model_inputs = tokenizer(["a b c", "d"], {
        text_pair: ["e f", "g h"],
        padding: true,
        truncation: true,
        return_tensor: false,
      });

      const expected = {
        input_ids: [
          [101, 1037, 1038, 1039, 102, 1041, 1042, 102],
          [101, 1040, 102, 1043, 1044, 102, 0, 0],
        ],
        token_type_ids: [
          [0, 0, 0, 0, 0, 1, 1, 1],
          [0, 0, 0, 1, 1, 1, 0, 0],
        ],
        attention_mask: [
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 0, 0],
        ],
      };

      compare(model_inputs, expected);
    },
    MAX_EXECUTION_TIME,
  );

  it(
    "should add token type ids if user requests them",
    async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/llama3-tokenizer-new");

      {
        // Without text pair
        const model_inputs = tokenizer("hello", {
          return_tensor: false,
          return_token_type_ids: true,
        });
        const expected = {
          input_ids: [128000, 15339],
          attention_mask: [1, 1],
          token_type_ids: [0, 0],
        };
        compare(model_inputs, expected);
      }

      {
        // With text pair
        const model_inputs = tokenizer("hello", {
          text_pair: "world",
          return_tensor: false,
          return_token_type_ids: true,
        });
        const expected = {
          input_ids: [128000, 15339, 128000, 14957],
          attention_mask: [1, 1, 1, 1],
          token_type_ids: [0, 0, 1, 1],
        };
        compare(model_inputs, expected);
      }
    },
    MAX_EXECUTION_TIME,
  );
});

describe("Edge cases", () => {
  it(
    "should not crash when encoding a very long string",
    async () => {
      let tokenizer = await AutoTokenizer.from_pretrained("Xenova/t5-small");

      let text = String.prototype.repeat.call("Hello world! ", 50000);
      let encoded = tokenizer(text);
      expect(encoded.input_ids.data.length).toBeGreaterThan(100000);
    },
    MAX_EXECUTION_TIME,
  );

  it("should not take too long", async () => {
    let tokenizer = await AutoTokenizer.from_pretrained("Xenova/all-MiniLM-L6-v2");

    let text = String.prototype.repeat.call("a", 50000);
    let token_ids = tokenizer.encode(text);
    compare(token_ids, [101, 100, 102]);
  }, 5000); // NOTE: 5 seconds

  it(
    "Special/added tokens with earlier partial matches",
    async () => {
      let tokenizer = await AutoTokenizer.from_pretrained("Xenova/gemini-nano");
      {
        let token_ids = tokenizer.encode("\n", { add_special_tokens: false });
        compare(token_ids, [108]);
      }
      {
        let token_ids = tokenizer.encode("\n\n", { add_special_tokens: false });
        compare(token_ids, [109]); // Should not be [108, 108]
      }
    },
    MAX_EXECUTION_TIME,
  );
});

describe("Extra decoding tests", () => {
  it(
    "should be able to decode the output of encode",
    async () => {
      let tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

      let text = "hello world!";

      // Ensure all the following outputs are the same:
      // 1. Tensor of ids: allow decoding of 1D or 2D tensors.
      let encodedTensor = tokenizer(text);
      let decoded1 = tokenizer.decode(encodedTensor.input_ids, { skip_special_tokens: true });
      let decoded2 = tokenizer.batch_decode(encodedTensor.input_ids, { skip_special_tokens: true })[0];
      expect(decoded1).toEqual(text);
      expect(decoded2).toEqual(text);

      // 2. List of ids
      let encodedList = tokenizer(text, { return_tensor: false });
      let decoded3 = tokenizer.decode(encodedList.input_ids, { skip_special_tokens: true });
      let decoded4 = tokenizer.batch_decode([encodedList.input_ids], { skip_special_tokens: true })[0];
      expect(decoded3).toEqual(text);
      expect(decoded4).toEqual(text);
    },
    MAX_EXECUTION_TIME,
  );
});

describe("Chat templates", () => {
  it("should generate a chat template", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/mistral-tokenizer-v1");

    const chat = [
      { role: "user", content: "Hello, how are you?" },
      { role: "assistant", content: "I'm doing great. How can I help you today?" },
      { role: "user", content: "I'd like to show off how chat templating works!" },
    ];

    const text = tokenizer.apply_chat_template(chat, { tokenize: false });

    expect(text).toEqual("<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]");

    const input_ids = tokenizer.apply_chat_template(chat, { tokenize: true, return_tensor: false });
    compare(input_ids, [1, 733, 16289, 28793, 22557, 28725, 910, 460, 368, 28804, 733, 28748, 16289, 28793, 28737, 28742, 28719, 2548, 1598, 28723, 1602, 541, 315, 1316, 368, 3154, 28804, 2, 28705, 733, 16289, 28793, 315, 28742, 28715, 737, 298, 1347, 805, 910, 10706, 5752, 1077, 3791, 28808, 733, 28748, 16289, 28793]);
  });

  it("should support multiple chat templates", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/c4ai-command-r-v01-tokenizer");

    // define conversation input:
    const conversation = [{ role: "user", content: "Whats the biggest penguin in the world?" }];
    // define documents to ground on:
    const documents = [
      { title: "Tall penguins", text: "Emperor penguins are the tallest growing up to 122 cm in height." },
      { title: "Penguin habitats", text: "Emperor penguins only live in Antarctica." },
    ];

    // render the RAG prompt as a string:
    const grounded_generation_prompt = tokenizer.apply_chat_template(conversation, {
      chat_template: "rag",
      tokenize: false,
      add_generation_prompt: true,

      documents,
      citation_mode: "accurate", // or "fast"
    });
    expect(grounded_generation_prompt).toEqual("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble\nThe instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.\n\n" + "# System Preamble\n## Basic Rules\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.\n\n" + "# User Preamble\n## Task and Context\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.\n\n## Style Guide\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|>" + "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|>" + "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>\nDocument: 0\ntitle: Tall penguins\ntext: Emperor penguins are the tallest growing up to 122 cm in height.\n\nDocument: 1\ntitle: Penguin habitats\ntext: Emperor penguins only live in Antarctica.\n</results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.\nFirstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.\nSecondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.\nThirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\nFinally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|>" + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
  });

  it("should support user-defined chat template", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/llama-tokenizer");

    const chat = [
      { role: "user", content: "Hello, how are you?" },
      { role: "assistant", content: "I'm doing great. How can I help you today?" },
      { role: "user", content: "I'd like to show off how chat templating works!" },
    ];

    // https://discuss.huggingface.co/t/issue-with-llama-2-chat-template-and-out-of-date-documentation/61645/3
    const chat_template = (
      "{% if messages[0]['role'] == 'system' %}" +
      "{% set loop_messages = messages[1:] %}" + // Extract system message if it's present
      "{% set system_message = messages[0]['content'] %}" +
      "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}" +
      "{% set loop_messages = messages %}" + // Or use the default system message if the flag is set
      "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}" +
      "{% else %}" +
      "{% set loop_messages = messages %}" +
      "{% set system_message = false %}" +
      "{% endif %}" +
      "{% if loop_messages|length == 0 and system_message %}" + // Special handling when only sys message present
      "{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}" +
      "{% endif %}" +
      "{% for message in loop_messages %}" + // Loop over all non-system messages
      "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}" +
      "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}" +
      "{% endif %}" +
      "{% if loop.index0 == 0 and system_message != false %}" + // Embed system message in first message
      "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}" +
      "{% else %}" +
      "{% set content = message['content'] %}" +
      "{% endif %}" +
      "{% if message['role'] == 'user' %}" + // After all of that, handle messages/roles in a fairly normal way
      "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}" +
      "{% elif message['role'] == 'system' %}" +
      "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}" +
      "{% elif message['role'] == 'assistant' %}" +
      "{{ ' '  + content.strip() + ' ' + eos_token }}" +
      "{% endif %}" +
      "{% endfor %}"
    )
      .replaceAll("USE_DEFAULT_PROMPT", true)
      .replaceAll("DEFAULT_SYSTEM_MESSAGE", "You are a helpful, respectful and honest assistant.");

    const text = tokenizer.apply_chat_template(chat, { tokenize: false, return_tensor: false, chat_template });

    expect(text).toEqual("<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]");

    // TODO: Add test for token_ids once bug in transformers is fixed.
  });

  it("should support default parameters", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/Meta-Llama-3.1-Tokenizer");

    // Example adapted from https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct#tool-use-with-transformers
    const chat = [
      { role: "system", content: "You are a bot that responds to weather queries." },
      { role: "user", content: "Hey, what's the temperature in Paris right now?" },
    ];
    const tools = [{ type: "function", function: { name: "get_current_temperature", description: "Get the current temperature at a location.", parameters: { type: "object", properties: { location: { type: "string", description: 'The location to get the temperature for, in the format "City, Country"' } }, required: ["location"] }, return: { type: "number", description: "The current temperature at the specified location in the specified units, as a float." } } }];

    {
      // `tools` unset (will default to `null`)
      const text = tokenizer.apply_chat_template(chat, { tokenize: false });
      expect(text).toEqual("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a bot that responds to weather queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHey, what's the temperature in Paris right now?<|eot_id|>");

      const input_ids = tokenizer.apply_chat_template(chat, { tokenize: true, return_tensor: false });
      compare(input_ids, [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11164, 430, 31680, 311, 9282, 20126, 13, 128009, 128006, 882, 128007, 271, 19182, 11, 1148, 596, 279, 9499, 304, 12366, 1314, 1457, 30, 128009]);
    }

    {
      // `tools` set
      const text = tokenizer.apply_chat_template(chat, { tools, tokenize: false });
      expect(text).toEqual('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a bot that responds to weather queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.\n\n{\n    "type": "function",\n    "function": {\n        "name": "get_current_temperature",\n        "description": "Get the current temperature at a location.",\n        "parameters": {\n            "type": "object",\n            "properties": {\n                "location": {\n                    "type": "string",\n                    "description": "The location to get the temperature for, in the format \\"City, Country\\""\n                }\n            },\n            "required": [\n                "location"\n            ]\n        },\n        "return": {\n            "type": "number",\n            "description": "The current temperature at the specified location in the specified units, as a float."\n        }\n    }\n}\n\nHey, what\'s the temperature in Paris right now?<|eot_id|>');

      const input_ids = tokenizer.apply_chat_template(chat, { tools, tokenize: true, return_tensor: false });
      compare(input_ids, [128000, 128006, 9125, 128007, 271, 13013, 25, 6125, 27993, 198, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11164, 430, 31680, 311, 9282, 20126, 13, 128009, 128006, 882, 128007, 271, 22818, 279, 2768, 5865, 11, 4587, 6013, 449, 264, 4823, 369, 264, 734, 1650, 449, 1202, 6300, 6105, 430, 1888, 11503, 279, 2728, 10137, 382, 66454, 304, 279, 3645, 5324, 609, 794, 734, 836, 11, 330, 14105, 794, 11240, 315, 5811, 836, 323, 1202, 907, 7966, 5519, 539, 1005, 7482, 382, 517, 262, 330, 1337, 794, 330, 1723, 761, 262, 330, 1723, 794, 341, 286, 330, 609, 794, 330, 456, 11327, 54625, 761, 286, 330, 4789, 794, 330, 1991, 279, 1510, 9499, 520, 264, 3813, 10560, 286, 330, 14105, 794, 341, 310, 330, 1337, 794, 330, 1735, 761, 310, 330, 13495, 794, 341, 394, 330, 2588, 794, 341, 504, 330, 1337, 794, 330, 928, 761, 504, 330, 4789, 794, 330, 791, 3813, 311, 636, 279, 9499, 369, 11, 304, 279, 3645, 7393, 13020, 11, 14438, 2153, 702, 394, 457, 310, 1173, 310, 330, 6413, 794, 2330, 394, 330, 2588, 702, 310, 5243, 286, 1173, 286, 330, 693, 794, 341, 310, 330, 1337, 794, 330, 4174, 761, 310, 330, 4789, 794, 330, 791, 1510, 9499, 520, 279, 5300, 3813, 304, 279, 5300, 8316, 11, 439, 264, 2273, 10246, 286, 457, 262, 457, 633, 19182, 11, 1148, 596, 279, 9499, 304, 12366, 1314, 1457, 30, 128009]);
    }
  });
});

describe("Decode ASR", () => {
  it(
    "should decode ASR outputs",
    async () => {
      const tokenizer = await WhisperTokenizer.from_pretrained("onnx-community/whisper-tiny.en_timestamped");

      const model_outputs = [
        {
          stride: [30, 0, 5],
          tokens: [50257n, 50362n, 8410n, 7283n, 0n, 2329n, 8410n, 7283n, 0n, 2094n, 470n, 1309n, 534n, 10625n, 307n, 10625n, 13n, 34668n, 345n, 531n, 9439n, 11n, 523n, 655n, 8410n, 7283n, 0n, 39134n, 16592n, 10625n, 0n, 9440n, 36n, 26751n, 0n, 25848n, 8410n, 7283n, 0n, 2773n, 661n, 4320n, 1943n, 981n, 345n, 821n, 8066n, 7765n, 510n, 290n, 670n, 1327n, 379n, 340n, 13n, 10528n, 318n, 5340n, 0n, 50256n],
          token_timestamps: [0, 0, 0, 3.78, 4.22, 5.34, 6.04, 6.56, 7, 7.92, 8.58, 8.58, 8.88, 9.14, 9.54, 9.94, 10.58, 11.38, 11.88, 12.42, 12.62, 13, 13.36, 13.64, 14.26, 14.76, 15.12, 15.4, 15.74, 16.12, 16.66, 17.14, 17.24, 17.24, 17.72, 18.38, 18.6, 19.38, 19.92, 22.66, 22.9, 23.24, 23.5, 24.14, 24.56, 24.7, 24.72, 24.94, 25.18, 25.54, 25.72, 26.02, 26.34, 26.44, 26.84, 27.04, 27.16, 27.54, 28.06, 29.92],
        },
        {
          stride: [30, 5, 5],
          tokens: [50257n, 50362n, 2773n, 661n, 4320n, 1943n, 981n, 345n, 821n, 8066n, 7765n, 510n, 290n, 670n, 1327n, 379n, 340n, 13n, 10528n, 318n, 5340n, 13n, 921n, 815n, 651n, 284n, 262n, 966n, 810n, 2687n, 2073n, 561n, 11238n, 290n, 345n, 821n, 407n, 8066n, 2245n, 612n, 13n, 1400n, 11n, 644n, 389n, 345n, 4953n, 329n, 30n, 2141n, 340n, 0n, 2329n, 466n, 340n, 0n, 3363n, 345n, 460n, 0n, 2329n, 466n, 340n, 0n, 50256n],
          token_timestamps: [0, 0, 0, 2.92, 3.24, 3.48, 4.14, 4.56, 4.7, 4.74, 4.92, 5.18, 5.54, 5.72, 6.04, 6.34, 6.46, 6.84, 7.04, 7.16, 7.54, 8.12, 10.16, 10.7, 10.9, 11.12, 11.24, 11.48, 11.84, 12.44, 12.82, 13.2, 13.46, 13.72, 14.06, 14.28, 14.34, 14.56, 14.8, 15.16, 15.9, 16.42, 16.82, 16.86, 17.02, 17.1, 17.22, 17.56, 18.06, 19.28, 19.62, 20.26, 21.96, 22.64, 24.28, 24.76, 25.18, 25.56, 25.78, 26.28, 27.12, 27.54, 27.82, 28.22, 29.48],
        },
        {
          stride: [23.7728125, 5, 0],
          tokens: [50257n, 50362n, 2329n, 466n, 340n, 0n, 3363n, 345n, 460n, 0n, 2329n, 466n, 340n, 0n, 1002n, 345n, 821n, 10032n, 286n, 3599n, 625n, 11n, 2245n, 3501n, 510n, 13n, 50256n],
          token_timestamps: [0, 0, 0, 2.44, 4.3, 5.04, 5.06, 5.56, 5.8, 6.32, 7.12, 7.56, 7.8, 8.9, 10.92, 12.96, 13.28, 13.28, 13.44, 13.72, 13.96, 14.84, 15.5, 16.06, 16.86, 17.88, 20.92],
        },
      ];

      const target = [
        " DO IT! Just DO IT! Don't let your dreams be dreams. Yesterday you said tomorrow, so just DO IT! MAKE YOUR dreams! COME TRUE! JUST DO IT! Some people dream success while you're gonna wake up and work hard at it. Nothing is impossible. You should get to the point where anyone else would quit and you're not gonna stop there. No, what are you waiting for? Do it! Just do it! Yes you can! Just do it! If you're tired of starting over, stop giving up.",
        {
          chunks: [
            { text: " DO", timestamp: [0.0, 3.78] },
            { text: " IT!", timestamp: [3.78, 4.24 /* 5.34 */] },
            { text: " Just", timestamp: [5.34, 6.04] },
            { text: " DO", timestamp: [6.04, 6.56] },
            { text: " IT!", timestamp: [6.56, 7.02 /* 7.92 */] },
            { text: " Don't", timestamp: [7.92, 8.58] },
            { text: " let", timestamp: [8.58, 8.88] },
            { text: " your", timestamp: [8.88, 9.14] },
            { text: " dreams", timestamp: [9.14, 9.54] },
            { text: " be", timestamp: [9.54, 9.94] },
            { text: " dreams.", timestamp: [9.94, 10.6 /* 11.38 */] },
            { text: " Yesterday", timestamp: [11.38, 11.88] },
            { text: " you", timestamp: [11.88, 12.42] },
            { text: " said", timestamp: [12.42, 12.62] },
            { text: " tomorrow,", timestamp: [12.62, 13.02 /* 13.36 */] },
            { text: " so", timestamp: [13.36, 13.64] },
            { text: " just", timestamp: [13.64, 14.26] },
            { text: " DO", timestamp: [14.26, 14.76] },
            { text: " IT!", timestamp: [14.76, 15.14 /* 15.4 */] },
            { text: " MAKE", timestamp: [15.4, 15.74] },
            { text: " YOUR", timestamp: [15.74, 16.12] },
            { text: " dreams!", timestamp: [16.12, 16.68 /* 17.14 */] },
            { text: " COME", timestamp: [17.14, 17.24] },
            { text: " TRUE!", timestamp: [17.24, 17.74 /* 18.38 */] },
            { text: " JUST", timestamp: [18.38, 18.6] },
            { text: " DO", timestamp: [18.6, 19.38] },
            { text: " IT!", timestamp: [19.38, 19.94 /* 22.66 */] },
            { text: " Some", timestamp: [22.66, 22.9] },
            { text: " people", timestamp: [22.9, 23.24] },
            { text: " dream", timestamp: [23.24, 23.5] },
            { text: " success", timestamp: [23.5, 24.14] },
            { text: " while", timestamp: [24.14, 24.56] },
            { text: " you're", timestamp: [24.56, 24.72] },
            { text: " gonna", timestamp: [24.72, 24.94] },
            { text: " wake", timestamp: [24.94, 25.18] },
            { text: " up", timestamp: [25.18, 25.54] },
            { text: " and", timestamp: [25.54, 25.72] },
            { text: " work", timestamp: [25.72, 26.04] },
            { text: " hard", timestamp: [26.04, 26.34] },
            { text: " at", timestamp: [26.34, 26.46] },
            { text: " it.", timestamp: [26.46, 26.86 /* 27.04 */] },
            { text: " Nothing", timestamp: [27.04, 27.16] },
            { text: " is", timestamp: [27.16, 27.54] },
            { text: " impossible.", timestamp: [27.54, 28.14 /* 30.16 */] },
            { text: " You", timestamp: [30.16, 30.7] },
            { text: " should", timestamp: [30.7, 30.9] },
            { text: " get", timestamp: [30.9, 31.12] },
            { text: " to", timestamp: [31.12, 31.24] },
            { text: " the", timestamp: [31.24, 31.48] },
            { text: " point", timestamp: [31.48, 31.84] },
            { text: " where", timestamp: [31.84, 32.44] },
            { text: " anyone", timestamp: [32.44, 32.82] },
            { text: " else", timestamp: [32.82, 33.2] },
            { text: " would", timestamp: [33.2, 33.46] },
            { text: " quit", timestamp: [33.46, 33.72] },
            { text: " and", timestamp: [33.72, 34.06] },
            { text: " you're", timestamp: [34.06, 34.34] },
            { text: " not", timestamp: [34.34, 34.56] },
            { text: " gonna", timestamp: [34.56, 34.8] },
            { text: " stop", timestamp: [34.8, 35.16] },
            { text: " there.", timestamp: [35.16, 35.92 /* 36.42 */] },
            { text: " No,", timestamp: [36.42, 36.84 /* 36.86 */] },
            { text: " what", timestamp: [36.86, 37.02] },
            { text: " are", timestamp: [37.02, 37.1] },
            { text: " you", timestamp: [37.1, 37.22] },
            { text: " waiting", timestamp: [37.22, 37.56] },
            { text: " for?", timestamp: [37.56, 38.08 /* 39.28 */] },
            { text: " Do", timestamp: [39.28, 39.62] },
            { text: " it!", timestamp: [39.62, 40.28 /* 41.96 */] },
            { text: " Just", timestamp: [41.96, 42.64] },
            { text: " do", timestamp: [42.64, 44.28] },
            { text: " it!", timestamp: [44.28, 44.78 /* 45.18 */] },
            { text: " Yes", timestamp: [45.18, 45.56] },
            { text: " you", timestamp: [45.56, 45.78] },
            { text: " can!", timestamp: [45.8, 46.34 /* 47.12 */] },
            { text: " Just", timestamp: [47.12, 47.56] },
            { text: " do", timestamp: [47.56, 47.8] },
            { text: " it!", timestamp: [47.8, 48.92 /* 50.92 */] },
            { text: " If", timestamp: [50.92, 52.96] },
            { text: " you're", timestamp: [52.96, 53.28] },
            { text: " tired", timestamp: [53.28, 53.44] },
            { text: " of", timestamp: [53.44, 53.72] },
            { text: " starting", timestamp: [53.72, 53.96] },
            { text: " over,", timestamp: [53.96, 54.86 /* 55.5 */] },
            { text: " stop", timestamp: [55.5, 56.06] },
            { text: " giving", timestamp: [56.06, 56.86] },
            { text: " up.", timestamp: [56.86, 57.9 /* 60.92 */] },
          ],
        },
      ];

      compare(
        tokenizer._decode_asr(model_outputs, {
          return_timestamps: "word",
          time_precision: 0.02,
          force_full_sequences: false,
        }),
        target,
        1e-2,
      );
    },
    MAX_EXECUTION_TIME,
  );

  it(
    "should handle overlapping edge case",
    async () => {
      const tokenizer = await WhisperTokenizer.from_pretrained("onnx-community/whisper-tiny.en_timestamped");

      const model_outputs = [
        {
          stride: [30, 0, 5],
          tokens: [50257n, 50362n, 8410n, 7283n, 0n, 2329n, 8410n, 7283n, 0n, 2094n, 470n, 1309n, 534n, 10625n, 307n, 10625n, 13n, 34668n, 11n, 345n, 531n, 9439n, 11n, 523n, 655n, 8410n, 7283n, 0n, 39134n, 16592n, 10560n, 3955n, 50n, 0n, 7102n, 5446n, 46n, 0n, 25848n, 8410n, 7283n, 0n, 2773n, 661n, 4320n, 1943n, 981n, 345n, 821n, 8066n, 7765n, 510n, 290n, 670n, 1327n, 379n, 340n, 13n, 10528n, 318n, 5340n, 13n, 50256n],
          token_timestamps: [0, 0, 0, 3.78, 4.22, 5.26, 6.04, 6.54, 7, 7.94, 8.58, 8.58, 8.88, 9.16, 9.54, 9.94, 10.6, 11.38, 11.88, 12.38, 12.44, 12.62, 13, 13.36, 13.64, 14.24, 14.74, 15.12, 15.4, 15.74, 16.1, 16.54, 16.54, 16.78, 17.08, 17.2, 17.36, 17.56, 18.08, 18.58, 19.38, 19.88, 22.54, 22.9, 23.24, 23.5, 24.14, 24.56, 24.7, 24.94, 24.94, 25.18, 25.54, 25.72, 26.04, 26.34, 26.46, 26.84, 27.04, 27.14, 27.54, 28.06, 29.92],
        },
        {
          stride: [30, 5, 5],
          tokens: [50257n, 50362n, 2773n, 661n, 4320n, 1943n, 981n, 345n, 821n, 8066n, 7765n, 510n, 290n, 670n, 1327n, 379n, 340n, 13n, 10528n, 318n, 5340n, 13n, 921n, 815n, 651n, 284n, 262n, 966n, 810n, 2687n, 2073n, 561n, 11238n, 290n, 345n, 821n, 407n, 8066n, 2245n, 612n, 13n, 1400n, 11n, 644n, 389n, 345n, 4953n, 329n, 30n, 2141n, 340n, 0n, 2329n, 466n, 340n, 0n, 3363n, 11n, 345n, 460n, 0n, 2329n, 466n, 340n, 0n, 50256n],
          token_timestamps: [0, 0, 0, 2.92, 3.24, 3.5, 4.14, 4.56, 4.7, 4.74, 4.92, 5.18, 5.54, 5.74, 6.04, 6.34, 6.46, 6.84, 7.04, 7.18, 7.56, 8.12, 9.68, 10.7, 10.88, 11.1, 11.24, 11.48, 11.82, 12.46, 12.82, 13.2, 13.46, 13.72, 14.08, 14.28, 14.34, 14.56, 14.82, 15.16, 15.72, 16.42, 16.82, 16.86, 17, 17.1, 17.2, 17.56, 18.06, 19.28, 19.6, 20.28, 21.96, 22.64, 24.28, 24.76, 25.18, 25.56, 25.56, 25.84, 26.36, 27.12, 27.54, 27.82, 28.16, 29.48],
        },
        {
          stride: [23.7728125, 5, 0],
          tokens: [50257n, 50362n, 2329n, 466n, 340n, 0n, 3363n, 345n, 460n, 0n, 2329n, 466n, 340n, 0n, 1002n, 534n, 15867n, 318n, 3599n, 625n, 11n, 2245n, 3501n, 510n, 13n, 50256n],
          token_timestamps: [0, 0, 0, 2.44, 4.3, 5.04, 5.06, 5.56, 5.8, 6.32, 7.12, 7.56, 7.8, 8.72, 10.04, 12.96, 13.3, 13.44, 13.72, 13.98, 14.86, 15.5, 16, 16.88, 17.76, 20.9],
        },
      ];

      const target = [
        " DO IT! Just DO IT! Don't let your dreams be dreams. Yesterday, you said tomorrow, so just DO IT! MAKE YOUR DRIMS! CONTRO! JUST DO IT! Some people dream success while you're gonna wake up and work hard at it. Nothing is impossible. You should get to the point where anyone else would quit and you're not gonna stop there. No, what are you waiting for? Do it! Just do it! Yes, you can! Just do it! If your tire is starting over, stop giving up.",
        {
          chunks: [
            { text: " DO", timestamp: [0, 3.78] },
            { text: " IT!", timestamp: [3.78, 4.24] },
            { text: " Just", timestamp: [5.26, 6.04] },
            { text: " DO", timestamp: [6.04, 6.54] },
            { text: " IT!", timestamp: [6.54, 7.02] },
            { text: " Don't", timestamp: [7.94, 8.58] },
            { text: " let", timestamp: [8.58, 8.88] },
            { text: " your", timestamp: [8.88, 9.16] },
            { text: " dreams", timestamp: [9.16, 9.54] },
            { text: " be", timestamp: [9.54, 9.94] },
            { text: " dreams.", timestamp: [9.94, 10.62] },
            { text: " Yesterday,", timestamp: [11.38, 11.9] },
            { text: " you", timestamp: [12.38, 12.44] },
            { text: " said", timestamp: [12.44, 12.62] },
            { text: " tomorrow,", timestamp: [12.62, 13.02] },
            { text: " so", timestamp: [13.36, 13.64] },
            { text: " just", timestamp: [13.64, 14.24] },
            { text: " DO", timestamp: [14.24, 14.74] },
            { text: " IT!", timestamp: [14.74, 15.14] },
            { text: " MAKE", timestamp: [15.4, 15.74] },
            { text: " YOUR", timestamp: [15.74, 16.1] },
            { text: " DRIMS!", timestamp: [16.1, 16.8] },
            { text: " CONTRO!", timestamp: [17.08, 17.58] },
            { text: " JUST", timestamp: [18.08, 18.58] },
            { text: " DO", timestamp: [18.58, 19.38] },
            { text: " IT!", timestamp: [19.38, 19.9] },
            { text: " Some", timestamp: [22.54, 22.9] },
            { text: " people", timestamp: [22.9, 23.24] },
            { text: " dream", timestamp: [23.24, 23.5] },
            { text: " success", timestamp: [23.5, 24.14] },
            { text: " while", timestamp: [24.14, 24.56] },
            { text: " you're", timestamp: [24.56, 24.94] },
            { text: " gonna", timestamp: [24.94, 24.94] },
            { text: " wake", timestamp: [24.94, 25.18] },
            { text: " up", timestamp: [25.18, 25.54] },
            { text: " and", timestamp: [25.54, 25.74] },
            { text: " work", timestamp: [25.74, 26.04] },
            { text: " hard", timestamp: [26.04, 26.34] },
            { text: " at", timestamp: [26.34, 26.46] },
            { text: " it.", timestamp: [26.46, 26.86] },
            { text: " Nothing", timestamp: [27.04, 27.18] },
            { text: " is", timestamp: [27.18, 27.56] },
            { text: " impossible.", timestamp: [27.56, 28.14] },
            { text: " You", timestamp: [29.68, 30.7] },
            { text: " should", timestamp: [30.7, 30.88] },
            { text: " get", timestamp: [30.88, 31.1] },
            { text: " to", timestamp: [31.1, 31.24] },
            { text: " the", timestamp: [31.24, 31.48] },
            { text: " point", timestamp: [31.48, 31.82] },
            { text: " where", timestamp: [31.82, 32.46] },
            { text: " anyone", timestamp: [32.46, 32.82] },
            { text: " else", timestamp: [32.82, 33.2] },
            { text: " would", timestamp: [33.2, 33.46] },
            { text: " quit", timestamp: [33.46, 33.72] },
            { text: " and", timestamp: [33.72, 34.08] },
            { text: " you're", timestamp: [34.08, 34.34] },
            { text: " not", timestamp: [34.34, 34.56] },
            { text: " gonna", timestamp: [34.56, 34.82] },
            { text: " stop", timestamp: [34.82, 35.16] },
            { text: " there.", timestamp: [35.16, 35.74] },
            { text: " No,", timestamp: [36.42, 36.84] },
            { text: " what", timestamp: [36.86, 37] },
            { text: " are", timestamp: [37, 37.1] },
            { text: " you", timestamp: [37.1, 37.2] },
            { text: " waiting", timestamp: [37.2, 37.56] },
            { text: " for?", timestamp: [37.56, 38.08] },
            { text: " Do", timestamp: [39.28, 39.6] },
            { text: " it!", timestamp: [39.6, 40.3] },
            { text: " Just", timestamp: [41.96, 42.64] },
            { text: " do", timestamp: [42.64, 44.28] },
            { text: " it!", timestamp: [44.28, 44.78] },
            { text: " Yes,", timestamp: [45.18, 45.56] },
            { text: " you", timestamp: [45.56, 45.84] },
            { text: " can!", timestamp: [45.8, 46.34] },
            { text: " Just", timestamp: [47.12, 47.56] },
            { text: " do", timestamp: [47.56, 47.8] },
            { text: " it!", timestamp: [47.8, 48.74] },
            { text: " If", timestamp: [50.04, 52.96] },
            { text: " your", timestamp: [52.96, 53.3] },
            { text: " tire", timestamp: [53.3, 53.44] },
            { text: " is", timestamp: [53.44, 53.72] },
            { text: " starting", timestamp: [53.72, 53.98] },
            { text: " over,", timestamp: [53.98, 54.88] },
            { text: " stop", timestamp: [55.5, 56] },
            { text: " giving", timestamp: [56, 56.88] },
            { text: " up.", timestamp: [56.88, 57.78] },
          ],
        },
      ];

      compare(
        tokenizer._decode_asr(model_outputs, {
          return_timestamps: "word",
          time_precision: 0.02,
          force_full_sequences: false,
        }),
        target,
        1e-2,
      );
    },
    MAX_EXECUTION_TIME,
  );
});
