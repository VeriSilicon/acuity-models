var url = window.location.href; 
var regx = /#(.+)/;
var json = url.match(regx)[1];
$.getJSON(json,function(result){
    $('#editor').val(JSON.stringify(result));
    $('#user-interface').hide();
    render_net(result);
    })
    .error(function() {alert(json + ' not found.');});
