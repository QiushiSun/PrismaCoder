#!/usr/bin/env node
/**
 * Render a TypeScript-generated visualization by:
 *   1. Loading compiled JS into an HTML page with common libs (d3, Chart.js)
 *   2. Opening in Puppeteer headless
 *   3. Screenshotting the result to PNG
 *
 * Usage: node ts_render.js <js_file> <png_output> [timeout_ms]
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

function loadLocalLib(name) {
    const candidates = [
        path.join(__dirname, 'node_modules', name),
        path.join(__dirname, '..', 'node_modules', name),
    ];
    for (const p of candidates) {
        if (fs.existsSync(p)) return p;
    }
    return null;
}

function readLibFile(libPath, subPath) {
    const full = path.join(libPath, subPath);
    if (fs.existsSync(full)) return fs.readFileSync(full, 'utf-8');
    return '';
}

async function render(jsFile, pngOutput, timeoutMs = 15000) {
    const jsCode = fs.readFileSync(jsFile, 'utf-8');

    // Load libraries from local node_modules
    const d3Path = loadLocalLib('d3');
    const chartPath = loadLocalLib('chart.js');
    const d3Script = d3Path ? readLibFile(d3Path, 'dist/d3.min.js') : '';
    const chartScript = chartPath ? readLibFile(chartPath, 'dist/chart.umd.js') : '';

    const html = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: white; width: 1200px; min-height: 800px; overflow: hidden; }
  #container, #chart-container, #app, #root, #main, #visualization {
    width: 100%; min-height: 600px;
  }
  canvas { max-width: 100%; }
  svg { max-width: 100%; }
</style>
</head><body>
<div id="container">
  <div id="chart-container"></div>
  <div id="app"></div>
  <div id="root"></div>
  <div id="main"></div>
  <div id="visualization"></div>
</div>
<script>${d3Script}</script>
<script>${chartScript}</script>
<script>
// Shim for common patterns in generated TS code
if (typeof module === 'undefined') { var module = { exports: {} }; }
if (typeof exports === 'undefined') { var exports = module.exports; }
if (typeof require === 'undefined') {
    var require = function(name) {
        if (name === 'd3' || name.startsWith('d3-')) return typeof d3 !== 'undefined' ? d3 : {};
        if (name === 'chart.js' || name === 'chart.js/auto') return typeof Chart !== 'undefined' ? { Chart: Chart } : {};
        if (name === 'three') return typeof THREE !== 'undefined' ? THREE : {};
        if (name === 'echarts') return typeof echarts !== 'undefined' ? echarts : {};
        console.warn('require() shimmed for:', name);
        return {};
    };
}
window.__RENDER_DONE__ = false;
window.__RENDER_ERROR__ = null;
window.onerror = function(msg) {
    window.__RENDER_ERROR__ = msg;
    window.__RENDER_DONE__ = true;
};
try {
${jsCode}
    setTimeout(function() { window.__RENDER_DONE__ = true; }, 200);
} catch(e) {
    window.__RENDER_ERROR__ = e.message || String(e);
    window.__RENDER_DONE__ = true;
}
</script>
</body></html>`;

    let browser;
    try {
        browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
                   '--disable-dev-shm-usage'],
        });
        const page = await browser.newPage();
        await page.setViewport({ width: 1200, height: 800 });

        page.on('pageerror', (err) => {
            // silently collect page errors
        });

        await page.setContent(html, { waitUntil: 'domcontentloaded', timeout: timeoutMs });

        // Wait for rendering to settle, with fallback
        try {
            await page.waitForFunction('window.__RENDER_DONE__ === true', { timeout: Math.min(timeoutMs, 10000) });
        } catch (_) {
            // Timeout waiting for __RENDER_DONE__ — take screenshot anyway
        }
        await new Promise(r => setTimeout(r, 800));

        const renderError = await page.evaluate(() => window.__RENDER_ERROR__);
        if (renderError) {
            console.error('RENDER_ERROR:' + renderError);
            process.exit(2);
        }

        // Find the best element to screenshot
        const bodySize = await page.evaluate(() => {
            const body = document.body;
            const svgs = document.querySelectorAll('svg');
            const canvases = document.querySelectorAll('canvas');
            return {
                hasSvg: svgs.length > 0,
                hasCanvas: canvases.length > 0,
                bodyH: body.scrollHeight,
                bodyW: body.scrollWidth,
            };
        });

        if (bodySize.hasSvg || bodySize.hasCanvas) {
            // Resize viewport to content
            await page.setViewport({
                width: Math.min(bodySize.bodyW || 1200, 2400),
                height: Math.min(bodySize.bodyH || 800, 2000),
            });
            await new Promise(r => setTimeout(r, 200));
        }

        await page.screenshot({ path: pngOutput, fullPage: true });
        console.log('OK');
        process.exit(0);

    } catch (e) {
        console.error('PUPPETEER_ERROR:' + (e.message || String(e)));
        process.exit(3);
    } finally {
        if (browser) await browser.close();
    }
}

const args = process.argv.slice(2);
if (args.length < 2) {
    console.error('Usage: node ts_render.js <js_file> <png_output> [timeout_ms]');
    process.exit(1);
}
render(args[0], args[1], parseInt(args[2]) || 15000);
