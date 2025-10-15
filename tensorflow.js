    import * as use from '@tensorflow-models/universal-sentence-encoder';
    import * as tf from '@tensorflow/tfjs';
    import fs from 'fs/promises';

    async function main() {
    const data = JSON.parse(await fs.readFile('./tempQA.json', 'utf8'));

    console.log("Loading USE model...");
    const model = await use.load();
    console.log("USE model loaded.");

    const qaWithVectors = [];

    for (const item of data) {
        // Vector cho question
        const questionEmbedding = await model.embed([item.question]);
        const questionVector = questionEmbedding.arraySync()[0];

        // Vector cho answer
        const answerEmbedding = await model.embed([item.answer]);
        const answerVector = answerEmbedding.arraySync()[0];

        qaWithVectors.push({
        question: item.question,
        questionVector,
        answer: item.answer,
        answerVector
        });
    }

    await fs.writeFile('./qa_with_use_vectors.json', JSON.stringify(qaWithVectors, null, 2), 'utf8');
    console.log("Done! Saved to qa_with_use_vectors.json");
    }

    main().catch(console.error);
