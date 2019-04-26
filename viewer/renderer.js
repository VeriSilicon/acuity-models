// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.
//

function build_description(layer, lid) {
    var desc = '';
    for (var key in layer['parameters']) {
        val = layer['parameters'][key];
        desc += '<p class="list-group-item-text">' + key + ':&nbsp;<span class="tip-val">' + val + '</span></p>';
    }
    if (desc == '') {
        desc = '<p class="list-group-item-text">No parameters</p>';
    }
    var lid_str = '<strong class="label-op">';
    var segs = lid.split('/');
    for (var i in segs) {
        var seg = segs[i];
        if (i > 0) {
            seg = '/' + seg;
        }
        lid_str += '<p class="list-group-item-text label-op">' + seg + '</p>';
    }
    lid_str += '</strong>';
    desc = '<div class="alert alert-success">' + lid_str + desc + '</div>';
    return desc;
}

var render_net = function(net) {
//var pathname = path.join(__dirname, net_file);

//console.log(net);

var g = new dagreD3.graphlib.Graph()
			.setGraph({rankdir: 'tb',ranksep:30,nodesep:10,edgesep:20,marginx:10,marginy:10});
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
            if (layer['parameters']['group_number'] == layer['parameters']['weights']
                && layer['parameters']['group_number'] > 1) {
                label = 'dw-conv';
                style = 'depthwise-convolution';
            } else {
                label = 'conv';
            }
            label += '(' + layer['parameters']['ksize_h'] + 'x' + layer['parameters']['ksize_w'] + ')'
            break;
        case 'deconvolution':
            label = 'deconv';
            label += '(' + layer['parameters']['ksize_h'] + 'x' + layer['parameters']['ksize_w'] + ')'
            break;
        case 'pooling':
            label = 'pool';
            label += '(' + layer['parameters']['ksize_h'] + 'x' + layer['parameters']['ksize_w'] + ')'
            break;
        case 'poolwithargmax':
            label += '(' + layer['parameters']['ksize_h'] + 'x' + layer['parameters']['ksize_w'] + ')'
            break;
        case 'fullconnect':
            label += '(' + layer['parameters']['weights'] + ')'
        default:
            break;
    }
    g.setNode(lid, {description: build_description(layer, lid), class: style, label: label});
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
    //.offset([15, 40])
	.direction('e')
    .html(function(d) {
        return g.node(d).description;
    });


inner.selectAll("g.node")
    .call(tips)
    .on('mouseover', tips.show)
    .on('mouseout', tips.hide);


/*
inner.selectAll("g.node")
  .attr("title", function(v) { return g.node(v).description; })
  .each(function(v) { $(this).tipsy({ gravity: "w", opacity: 1, html: true }); });
*/

//var zoom = d3.zoom().on("zoom", function() {
//    inner.attr("transform", "translate(" + d3.event.transform.x + ',' + d3.event.transform.y + ")" +
//            "scale(" + d3.event.transform.k + ")");
	//svg.attr("transform", "translate(" + d3.event.transform.x + ',' + d3.event.transform.y + ")" +
    //        "scale(" + d3.event.transform.k + ")");
//});
//svg.call(zoom);
// var initialScale = 1;

// svg.attr('width', g.graph().width * initialScale + 100);
// svg.attr('height', g.graph().height * initialScale + 150);
// inner.attr("transform", "translate(15,20)" + "scale(" + initialScale + ")");

var  bbox = svg.node().getBBox();

svg.attr("x", bbox.x);
svg.attr("y", bbox.y);
svg.attr("width", bbox.width);
svg.attr("height", bbox.height);
svg.attr("viewBox", bbox.x + " " + bbox.y + " " +  bbox.width + " " + bbox.height);

// debug code to view bbox of svg canvas
// var rect = svg.append("rect")
// .attr("x", bbox.x)
// .attr("y", bbox.y)
// .attr("width", bbox.width)
// .attr("height", bbox.height)
// .style("fill", "#ccc")
// .style("fill-opacity", ".3")
// .style("stroke", "#666")
// .style("stroke-width", "1.5px");

//progress.attr('aria-valuenow', '100')
//progress.attr('style', 'width: 100%')

var net_name = d3.select("#net-name").text(net['MetaData']['Name']);

}

