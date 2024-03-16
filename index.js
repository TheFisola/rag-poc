const express = require("express");
const app = express();
const OpenAI = require("openai");
const { createClient } = require("@supabase/supabase-js");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { CheerioWebBaseLoader } = require("langchain/document_loaders/web/cheerio");
require("dotenv").config();

app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

app.post("/embed", async (req, res) => {
  try {
    await generateEmbeddings();
    res.status(200).json({ message: "Successfully Embedded" });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: "Error occurred",
    });
  }
});

app.post("/query", async (req, res) => {
  try {
    const { query } = req.body;
    const result = await handleQuery(query);
    res.status(200).json(result);
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: "Error occurred",
    });
  }
});

async function generateEmbeddings() {
    const loader = new CheerioWebBaseLoader(
      "https://www.inboxpurge.com/faq"
    );
    const docs = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const chunks = await textSplitter.splitDocuments(docs);

    const promises = chunks.map(async (chunk) => {
      const cleanChunk = chunk.pageContent.replace(/\n/g, " ");

      const embeddingResponse = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: cleanChunk,
      });

      const [{ embedding }] = embeddingResponse.data;

      const { error } = await supabase.from("documents").insert({
        content: cleanChunk,
        embedding,
      });

      if (error) {
        throw error;
      }
    });

    await Promise.all(promises);
}

async function handleQuery(query) {
  const input = query.replace(/\n/g, " ");

  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input,
  });

  const [{ embedding }] = embeddingResponse.data;

  const { data: documents, error } = await supabase.rpc("match_documents", {
    query_embedding: embedding,
    match_threshold: 0.5,
    match_count: 10,
  });

  if (error) throw error;

  let contextText = "";

  contextText += documents
    .map((document) => `${document.content.trim()}---\n`)
    .join("");

  const messages = [
    {
      role: "system",
      content: `You are a representative that is very helpful when it comes to talking about InboxPurge, Only ever answer
    truthfully and be as helpful as you can!`,
    },
    {
      role: "user",
      content: `Context sections: "${contextText}" Question: "${query}" Answer as simple text:`,
    },
  ];

  const completion = await openai.chat.completions.create({
    messages,
    model: "gpt-4",
    temperature: 0.8,
  });

  return completion.choices[0].message.content;
}

app.listen("3035", () => {
  console.log("App is running");
});
