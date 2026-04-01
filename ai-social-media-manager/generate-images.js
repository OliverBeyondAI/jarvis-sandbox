import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY || GEMINI_API_KEY === "your_gemini_api_key_here") {
  console.error("Error: Set a valid GEMINI_API_KEY in .env");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

const imagePrompts = [
  {
    filename: "hero-relaxing-business-owner.png",
    prompt:
      "A photorealistic wide-angle hero image of a confident small business owner relaxing in a modern office, leaning back in their chair with a coffee, smiling contentedly. Multiple holographic social media dashboards float in the air around them showing analytics, scheduled posts, and engagement metrics — all managed by AI. Warm ambient lighting, clean modern workspace, soft purple and blue accent lighting. Professional marketing photo style, 16:9 aspect ratio.",
  },
  {
    filename: "dashboard-analytics-mockup.png",
    prompt:
      "A sleek, modern social media analytics dashboard displayed on a large monitor screen. The dashboard shows colorful charts and graphs: follower growth line chart trending upward, engagement rate donut chart, posting schedule calendar, top performing posts grid, and audience demographics bar chart. Dark theme UI with purple and blue accent colors, clean typography, professional SaaS product screenshot style. Numbers show impressive growth metrics like +247% engagement and 12K new followers.",
  },
  {
    filename: "before-after-comparison.png",
    prompt:
      "A split-screen before-and-after comparison of a business social media feed. LEFT side labeled 'Before' shows a messy, inconsistent Instagram-style feed with poor quality images, irregular posting, low engagement counts (2-5 likes), and no cohesive branding — slightly desaturated and dull. RIGHT side labeled 'After AI Management' shows the same account transformed: cohesive visual branding, professional photos, consistent color palette in purple tones, high engagement counts (500+ likes, dozens of comments), verified badge, growing follower count. Clean mockup style with subtle purple gradient background.",
  },
];

async function generateImage(prompt, filename) {
  console.log(`Generating: ${filename}...`);

  try {
    const response = await ai.models.generateImages({
      model: "imagen-4.0-generate-001",
      prompt: prompt,
      config: {
        numberOfImages: 1,
      },
    });

    const image = response.generatedImages?.[0]?.image;
    if (!image?.imageBytes) {
      throw new Error("No image data in response");
    }

    const outputPath = path.join(__dirname, filename);
    const buffer = Buffer.from(image.imageBytes, "base64");
    fs.writeFileSync(outputPath, buffer);
    console.log(`Saved: ${filename} (${(buffer.length / 1024).toFixed(1)} KB)`);
    return filename;
  } catch (err) {
    console.error(`Failed to generate ${filename}: ${err.message}`);
    throw err;
  }
}

async function main() {
  console.log("AI Social Media Manager — Image Generator");
  console.log("==========================================");
  console.log("Using Gemini API with Imagen 4.0 (imagen capabilities)\n");

  const results = [];

  for (const { prompt, filename } of imagePrompts) {
    try {
      const saved = await generateImage(prompt, filename);
      results.push(saved);
    } catch {
      console.error(`Skipping ${filename} due to error.\n`);
    }
  }

  console.log(`\nDone! Generated ${results.length}/${imagePrompts.length} images.`);
  if (results.length > 0) {
    console.log("Files:", results.join(", "));
  }
}

main();
