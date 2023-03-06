import fs from 'fs';
import path from 'path';
import { URL } from 'url';
import tf from '@tensorflow/tfjs-node';
import canvas from 'canvas';

const { Canvas, loadImage } = canvas;
const __dirname = new URL('.', import.meta.url).pathname;

const DEFAULT_OPTIONS = {
  debug: process.env.NODE_ENV === 'development',
  modelPath: 'file://' + path.join(__dirname, '/model/model.json'),
  minScore: 0.30,
  maxResults: 50,
  iouThreshold: 0.5,
  outputNodes: ['output1', 'output2', 'output3'],
  blurNude: true,
  blurRadius: 20,
};

const LABELS = [
  'exposed anus',
  'exposed armpits',
  'belly',
  'exposed belly',
  'buttocks',
  'exposed buttocks',
  'female face',
  'male face',
  'feet',
  'exposed feet',
  'breast',
  'exposed breast',
  'vagina',
  'exposed vagina',
  'male breast',
  'exposed male breast',
];

// Define objects from labels
const COMPOSITE = {
  person: [6, 7],
  sexy: [1, 2, 3, 4, 8, 9, 10, 15],
  nude: [0, 5, 11, 12, 13],
};

const blur = ({ imgCanvas, options, left = 0, top = 0, width = 0, height = 0 }) => {
  if (!imgCanvas) return;

  const blurCanvas = new Canvas(width / options.blurRadius, height / options.blurRadius);
  const blurCtx = blurCanvas.getContext('2d');
  if (!blurCtx) return;

  blurCtx.imageSmoothingEnabled = true;
  blurCtx.drawImage(imgCanvas, left, top, width, height, 0, 0, width / options.blurRadius, height / options.blurRadius);

  const canvasCtx = imgCanvas.getContext('2d');
  canvasCtx.drawImage(blurCanvas, left, top, width, height);
};

const rect = ({ imgCanvas, x = 0, y = 0, width = 0, height = 0, radius = 8, lineWidth = 2, color = 'white', title = '', font = '16px "Segoe UI"' }) => {
  if (!imgCanvas) return;

  const ctx = imgCanvas.getContext('2d');
  if (!ctx) return;

  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.fillText(title, x + 4, y - 4);
};


class NNNudeNet {

  constructor(opts) {
    this.options = Object.assign(this, DEFAULT_OPTIONS, opts);
    this.output = this.options.outputNodes
  }

  async load() {
    try {
      this.model = await tf.loadGraphModel(this.options.modelPath);
    } catch (err) {
      console.error(`Error loading model from path: ${this.options.modelPath}\n\n${err}\n`)
      process.exit(-1);
    }
  }

  async predict(input, outputPath) {
    const t = {}
    t.input = this.__readImage(input);

    [t.boxes, t.scores, t.classes] = await this.model.executeAsync(t.input, this.output);
    const dataAnalysis = await this.__processImageAnalysis(t.boxes, t.scores, t.classes, t.input);
    Object.keys(t).forEach((tensor) => tf.dispose(t[tensor]));

    const output = outputPath || this.options.outputPath;
    if (output) {
      await this.__saveOutputImage(input, output, dataAnalysis);
    }

    return dataAnalysis;
  }

  __readImage(filePath) {
    if (!filePath || !fs.existsSync(filePath)) {
      console.error('File not found in path:', filePath);
      return null;
    }

    const data = fs.readFileSync(filePath);
    const bufferT = tf.node.decodeImage(data, 3);
    const expandedT = tf.expandDims(bufferT, 0);
    const imageT = tf.cast(expandedT, 'float32');
    imageT['file'] = filePath;

    tf.dispose([expandedT, bufferT]);

    return imageT;
  }

  async __processImageAnalysis(boxesT, scoresT, classesT, inputT) {
    const boxes = await boxesT.array();
    const scores = await scoresT.data();
    const classes = await classesT.data();

    const nmsT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, this.options.maxResults, this.options.iouThreshold, this.options.minScore);
    const nms = await nmsT.data();
    tf.dispose(nmsT);

    const parts = [];
    for (const i in nms) {
      const id = parseInt(i);
      parts.push({
        score: scores[i],
        id: classes[id],
        class: LABELS[classes[id]],
        box: [ // convert box from x0,y0,x1,y1 to x,y,width,heigh
          Math.trunc(boxes[0][id][0]),
          Math.trunc(boxes[0][id][1]),
          Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
          Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
        ],
      });
    }

    const result = {
      input: { file: inputT.file, width: inputT.shape[2], height: inputT.shape[1] },
      person: parts.filter((a) => COMPOSITE.person.includes(a.id)).length > 0,
      sexy: parts.filter((a) => COMPOSITE.sexy.includes(a.id)).length > 0,
      nude: parts.filter((a) => COMPOSITE.nude.includes(a.id)).length > 0,
      parts,
    };

    return result;
  }

  async __saveOutputImage(inputImage, outputPath, data) {
    if (!data) return;
    const options = this.options;

    return new Promise(async (resolve) => {
      const original = await loadImage(inputImage);
      const c = new Canvas(original.width, original.height);
      const ctx = c.getContext('2d');
      ctx.drawImage(original, 0, 0, c.width, c.height);

      for (const obj of data.parts) {
        if (COMPOSITE.nude.includes(obj.id) && options.blurNude) {
          blur({
            imgCanvas: c, options: options, left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3],
          });
        }
        if (options.debug) {
          rect({
            imgCanvas: c, x: obj.box[0], y: obj.box[1], width: obj.box[2], height: obj.box[3], title: `${Math.round(100 * obj.score)}% ${obj.class}`,
          });
        }
      }
      const out = fs.createWriteStream(outputPath);
      out.on('finish', () => {
        resolve(true);
      });
      out.on('error', (err) => {
        console.error('Could not create an image', outputPath, err);
        resolve(true);
      });
      const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
      stream.pipe(out);
    });
  }
}
