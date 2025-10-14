// prepare_embeddings_offline.js
import fs from "fs/promises";
import { pipeline } from "@xenova/transformers";

const MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"; // multilingual
const INPUT_FILE = "./tempQA.json";        // file JSON gốc
const OUTPUT_FILE = "./qa_with_vectors.json"; // file output

async function main() {
  console.log("Loading embedder model...");
  const embedder = await pipeline("feature-extraction", MODEL);
  console.log("Embedder ready.");

  // Load Q&A JSON
  const data = JSON.parse(await fs.readFile(INPUT_FILE, "utf8"));
  const qaWithVectors = [];

  console.log(`Computing embeddings for ${data.length} items...`);
  for (const item of data) {
    const out = await embedder(item.question, { pooling: "mean", normalize: true });
    qaWithVectors.push({
      question: item.question,
      answer: item.answer,
      vector: Array.from(out.data),
    });
  }

  // Save output file
  await fs.writeFile(OUTPUT_FILE, JSON.stringify(qaWithVectors, null, 2), "utf8");
  console.log(`Done! Embeddings saved to ${OUTPUT_FILE}`);
}
main().catch(err => console.error(err));
