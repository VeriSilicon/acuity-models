// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.
//

function build_description(layer, lid) {
    var desc = '';
    for (var key in layer['parameters']) {
        val = layer['parameters'][key];
        desc += '<p class="list-group-item-text">' + key + ':&nbsp;' + val + '</p>';
    }
    if (desc == '') {
        desc = '<p class="list-group-item-text">No parameters</p>';
    }
    desc = '<p class="list-group-item-text label-op"><strong>' + lid + '</strong></p>' + desc;
    desc = '<div class="alert alert-success">' + desc + '</div>';
    return desc;
}

var render_net = function(net) {
//var pathname = path.join(__dirname, net_file);

//console.log(net);

var g = new dagreD3.graphlib.Graph().setGraph({});
//var progress = d3.select('#parse-progress-bar');
//progress.attr('aria-valuenow', '0')
//progress.attr('style', 'width: 0%')

for (var lid in net.Layers) {
    layer = net.Layers[lid];
    var style = ''
    switch (layer['op']) {
        case 'relu':
        case 'prelu':
        case 'leakyrelu':
        case 'relun':
        case 'softmax':
        case 'dropout':
            style = 'activation';
            break;
        case 'reshape':
        case 'permute':
        case 'concat':
        case 'split':
        case 'eltwise':
        case 'localresponsenormalization':
        case 'l2normalizescale':
        case 'l2normalize':
        case 'batchnormalize':
        case 'multiply':
            style = 'no-batch-op';
            break;
        default:
            style = layer['op'];
            break;
    }
    if (layer['op'] == 'localresponsenormalization') {
        layer['op'] = 'lrn';
    }
    var label = layer['op'];
    switch (layer['op']) {
        case 'convolution':
        case 'deconvolution':
        case 'pooling':
        case 'poolwithargmax':
            label += '(' + layer['parameters']['ksize_h'] + 'x' + layer['parameters']['ksize_w'] + ')'
            break;
        case 'fullconnect':
            label += '(' + layer['parameters']['weights'] + ')'
        default:
            break;
    }
    g.setNode(lid, {description: build_description(layer, lid), class: style, label: label, width: 100, height: 20});
    //console.log(lid)
}
//progress.attr('aria-valuenow', '70')
//progress.attr('style', 'width: 70%')
//console.log(g)

var regx = /@(.+):/;
for (var lid in net.Layers) {
    layer = net.Layers[lid];
    layer['inputs'].forEach(function(tensor,idx,ar){
        var source = tensor.match(regx)[1];
        g.setEdge(source, lid, {arrowhead: 'undirected'});
        //console.log(source, lid);
        //console.log(g.edge(source, lid));
    });
}

//console.log(g)
var render = new dagreD3.render();

var svg = d3.select("#render-zone");
//svg.html("");
var inner = svg.select('#render-graph')
if (inner.empty()) {
    inner = svg.append("g")
            .attr('id', 'render-graph');
} else {
    inner.html("");
}

render(inner, g);

var tips = d3.tip()
    .attr('class', 'layer-tip')
    .offset([15, 40])
    .html(function(d) {
        return g.node(d).description;
    });


inner.selectAll("g.node")
    .call(tips)
    .on('mouseover', tips.show)
    .on('mouseout', tips.hide);

//inner.selectAll("g.node")
//     .attr('title', function(v) { return g.node(v).description })
//     .attr('data-toggle', 'tooltip');
//     //.attr('data-placement', 'right')
//$(function () { $("[data-toggle='tooltip']").tooltip(); });

//var zoom = d3.zoom().on("zoom", function() {
//    inner.attr("transform", "translate(" + d3.event.translate + ")" +
//            "scale(" + d3.event.scale + ")");
//    //svg.attr("transform", "translate(" + d3.event.translate + ")" +
//    //        "scale(" + d3.event.scale + ")");
//});
//svg.call(zoom);
var initialScale = 2;

svg.attr('width', g.graph().width * initialScale + 50);
svg.attr('height', g.graph().height * initialScale + 150);
inner.attr("transform", "translate(5,20)" + "scale(" + initialScale + ")");

//progress.attr('aria-valuenow', '100')
//progress.attr('style', 'width: 100%')

var net_name = d3.select("#net-name").text(net['MetaData']['Name']);

}

