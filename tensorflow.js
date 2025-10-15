    import * as use from '@tensorflow-models/universal-sentence-encoder';
    import * as tf from '@tensorflow/tfjs';
    import fs from 'fs/promises';

    async function main() {
    // 1. Load JSON
    const data = JSON.parse(await fs.readFile('./merged_qa.json', 'utf8'));

    // 2. Load Universal Sentence Encoder
    console.log("Loading USE model...");
    const model = await use.load();
    console.log("USE model loaded.");

    const qaWithVectors = [];

    for (const item of data) {
        // 3. Tính vector cho question
        const embeddings = await model.embed([item.data]);
        const vector = embeddings.arraySync()[0]; // vector dạng Array

        qaWithVectors.push({
        data: item.data,
        vector
        });
    }

    // 4. Lưu ra file JSON mới
    await fs.writeFile('./qa_with_use_vectors.json', JSON.stringify(qaWithVectors, null, 2), 'utf8');
    console.log("Done! Saved to qa_with_use_vectors.json");
    }

    main().catch(console.error);
