var url = window.location.href; 
var regx = /#(.+)/;
var json = url.match(regx)[1];
$.getJSON(json,function(result){
    //$('#editor').val(JSON.stringify(result));
    $('#user-interface').hide();
    render_net(result);
    })
    .error(function() {
		var progress_bar = $('#progress-bar');
		if (progress_bar) {
			progress_bar.attr("class", 'progress-bar progress-bar-danger')
				.html("File not found");
		}
		
		alert(json + ' not found.');
	});
