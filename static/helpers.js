function getRndInteger(min, max) {
	return Math.floor(Math.random() * (max - min) ) + min;
}

function getRndFloat(min, max) {
	return Math.random() * (max - min) + min;
}

function getRndFloatG(min, max, v) {
	return randomG(v) * (max - min) + min;
}

function randomG(v){ 
    var r = 0;
    for(var i = v; i > 0; i --){
        r += Math.random();
    }
    return r / v;
}

function average(data){
	var sum = data.reduce(function(sum, value){
		return sum + value;
	}, 0);
  
	var avg = sum / data.length;
	return avg;
}

function standardDeviation(values){
	var avg = average(values);
	
	var squareDiffs = values.map(function(value){
		var diff = value - avg;
		var sqrDiff = diff * diff;
		return sqrDiff;
	});
	
	var avgSquareDiff = average(squareDiffs);
  
	var stdDev = Math.sqrt(avgSquareDiff);
	return stdDev;
}

function normalize(values){
	const std_dev = standardDeviation(values);
	const avg = average(values);
	const normalized_values = values.map(function(value){
		return (value - avg)/std_dev;
	});
	return normalized_values;
}
