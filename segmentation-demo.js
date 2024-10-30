function info(message) {
    const info = document.getElementById("infoPane");
    // Append message paragraph to info element.
    const p = document.createElement("p");
    p.textContent = message;
    info.appendChild(p);
    // Scroll to bottom of info element.
    info.scrollTop = info.scrollHeight;
}

async function decode(imageURL, boundingBox) {

    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;

    // Convert the bounding box according to the processed image size.
    const ratio = processImageWidth / originalImageWidth;
    const box = boundingBox.map(x => Math.round(x * ratio));

    const response = await axios.post(`${apiBaseUrl}/flosam/seg-cap`, {
        image_url: imageURL,
        box: box,
    });
    const data = response.data;

    const segment = data.result.segmentation.segment.map(([x, y]) => [x / ratio, y / ratio]);;

    const description = data.result.description;
 
    return { segment, description };
}

async function drawPolygon(selection = null, polygon) {
    if (!selection) {
        selection = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "type": "Annotation",
            "body": [],
            "target": {},
        }
    } else {
        anno.cancelSelected();
    }
    const newSelector = {
        "type": "SvgSelector",
    };
    let svg = '<svg><polygon points="';
    for (const [x, y] of polygon) {
        svg += `${x},${y} `;
    }
    svg += '"/></svg>';
    newSelector.value = svg;
    selection.target.selector = newSelector;
    // Generate a random ID for the new annotation
    selection.id = Math.random().toString(36).substring(2, 15);
    
    anno.addAnnotation(selection);
    return selection.id;
}

async function segment(selection) {
    const loader = document.getElementById("loaderPane");
    loader.style.display = "flex";
    info("Segmenting in bounding box...");

    // Get the bounding box of the selection
    const selectorValue = selection.target.selector.value;  // In "xywh=pixel:0,0,100,100" format.
    const xywh = selectorValue.split('=')[1].split(':').pop().split(',').map(Number);
    const boundingBox = xywh.map((x, i) => i < 2 ? x : x + xywh[i - 2]).flat();
    
    const result = await decode(imageUrl, boundingBox);
    const annoID = await drawPolygon(selection, result.segment);
    info(`Polygon (size ${result.segment.length}) created`);
    annoDescriptions[annoID] = result.description;
    info(`Description: ${result.description}`);
    loader.style.display = "none";
}

const apiBaseUrl = 'http://ec2-13-210-199-164.ap-southeast-2.compute.amazonaws.com:8000';

const imageList = {
    'Buddha Year 5': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54MGYC39W9MZ32YD4GJ59D',
    'Schist Bodhisattva Head - Front': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HGY0E51T8B85A0023YNHN2R9',
    'Shakyamuni and Maitreya': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HGY2X02VBKQKW21J5AXKVT2Z',
    'Kanishka Casket': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54F7AZ3NXRBR5C3TQHNGHR',
    'Inscribed Stele': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54KWVM6DYFWGYZ5CECPKHN',
    'Bar Photo': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HGWC2EGD0VV6QTT3X91V9Z1H',
    'Guanyin Statue': 'https://puam-loris.aws.princeton.edu/loris/y1950-66_DET.jp2',
    'IR Photo': 'https://iiif.io/api/image/3.0/example/reference/918ecd18c2592080851777620de9bcb5-gottingen',
    'John Dee': 'https://iiif.io/api/image/3.0/example/reference/421e65be2ce95439b3ad6ef1f2ab87a9-dee-natural',
    'Book Page': 'https://iiif.io/api/image/3.0/example/reference/59d09e6773341f28ea166e9f3c1e674f-gallica_ark_12148_bpt6k1526005v_f20',
    'Whistlers_Mother': 'https://iiif.io/api/image/3.0/example/reference/329817fc8a251a01c393f517d8a17d87-Whistlers_Mother',
    'Old train photo': 'https://images.prov.vic.gov.au/loris/2164%2F1297%2F05%2Fimages%2F1%2Ffiles%2F12800-00001-000170-280.tif',
    'Gichi Nala': 'https://heidicon.ub.uni-heidelberg.de/iiif/2/23736792%3A960606',
};

let imageBase;

// Use the query string `image` as the image base URL. If not provided, use the default image.
const urlParams = new URLSearchParams(window.location.search);
const imageParam = urlParams.get('image');
if (imageParam) {
    imageBase = imageParam;
} else {
    imageBase = imageList['Buddha Year 5'];
}

const processImageWidth = 1024;
const imageUrl = `${imageBase}/full/${processImageWidth},/0/default.jpg`;

const annoDescriptions = {};

const viewer = OpenSeadragon({
    id: "osdContainer",
    visibilityRatio: 1,
    crossOriginPolicy: false,
    prefixUrl: "node_modules/openseadragon/build/openseadragon/images/",
    tileSources: `${imageBase}/info.json`,
});



const anno = OpenSeadragon.Annotorious(viewer, {
    allowEmpty: true,
    disableEditor: true,
});

anno.on('createSelection', function(selection) {
    segment(selection);
});

const tooltip = document.createElement('div');
tooltip.className = 'anno-tooltip';
Object.assign(tooltip.style, {
    display: 'none',
    position: 'absolute',
    backgroundColor: 'white',
    color: '#595959',
    fontSize: '0.8rem',
    padding: '0.5em',
    boxShadow: '0 3px 10px rgb(0 0 0 / 0.2)',
    maxWidth: '300px'
});
document.body.appendChild(tooltip);

anno.on('mouseEnterAnnotation', function(annotation, element) {
    tooltip.textContent = annoDescriptions[annotation.id];
    let position = element.getBoundingClientRect();
    tooltip.style.top = `${position.top + 10}px`;
    tooltip.style.left = `${position.left + 10}px`;
    tooltip.style.display = 'block';
});

anno.on('mouseLeaveAnnotation', function(annotation, element) {
    tooltip.style.display = 'none';
    tooltip.innerHTML = '';
});

const segAllFSBtn = document.getElementById('segAllFS');
segAllFSBtn.addEventListener('click', async function() {
    const loader = document.getElementById("loaderPane");
    loader.style.display = "flex";
    info("Segmenting the whole image...");

    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;

    // Convert the bounding box according to the processed image size.
    const ratio = processImageWidth / originalImageWidth;

    const response = await axios.post(`${apiBaseUrl}/flosam/seg-cap-all`, {
        image_url: imageUrl,
    });
    const data = response.data;

    for (const item of data.result) {
        const polygon = item.segment;
        const segment = polygon.map(([x, y]) => [x / ratio, y / ratio]);
        const description = item.label;
        const annoID = await drawPolygon(null, segment);
        annoDescriptions[annoID] = description;
    }
    info(`Segmentation completed`);
    loader.style.display = "none";
});
