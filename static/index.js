$("#linear-regression").click(async function () {
	$("#example-title").text("Linear Regression");
	const num_house = 160
	
	// Generate Demo Data
	const plot_houses = [];
	const plot_houses_train = [];
	const plot_houses_test = [];
	const train_house_sizes = [];
	const train_house_prices = [];
	const test_house_sizes = [];
	const test_house_prices = [];
	const num_train_samples = Math.floor(num_house * 0.7)
	for(var i = 0; i < num_house; i++){
		const house_size = getRndInteger(1000, 3500);
		const house_price = house_size * 100.0 + getRndInteger(20000, 70000);

		plot_houses.push({x: house_size, y: house_price});
		if (i < num_train_samples) {
			plot_houses_train.push({x: house_size, y: house_price});

			train_house_sizes.push(house_size);
			train_house_prices.push(house_price);
		} else {
			plot_houses_test.push({x: house_size, y: house_price});

			test_house_sizes.push(house_size);
			test_house_prices.push(house_price);
		}
	}

	// Plot Data
	tfvis.render.scatterplot(
		{ name: 'Houses', tab: 'Charts' },
		{ values: [plot_houses], series: ['houses']},
		{
			width: 400,
			height: 300,
			xAxisDomain: [500, 4000],
			yAxisDomain: [100000, 450000],
			xLabel: 'Size',
			yLabel: 'Price'
		});
	
	// Plot Train/Test Split
	tfvis.render.scatterplot(
		{ name: 'Train/Test Split', tab: 'Charts' },
		{ values: [plot_houses_train, plot_houses_test], series: ['Training data', 'Testing data'] },
		{
			width: 400,
			height: 300,
			xAxisDomain: [500, 4000],
			yAxisDomain: [100000, 450000],
			seriesColors: ['pink', 'green'],
			xLabel: 'Size',
			yLabel: 'Price'
		});
	
	const train_tensors = {
		houseSizes: tf.tensor2d(train_house_sizes, [train_house_sizes.length, 1]),
		housePrices: tf.tensor2d(train_house_prices, [train_house_prices.length, 1])
	};
	const test_tensors = {
		houseSizes: tf.tensor2d(test_house_sizes, [test_house_sizes.length, 1]),
		housePrices: tf.tensor2d(test_house_prices, [test_house_prices.length, 1])
	};

	// define model structure
	const model = tf.sequential();
	// housePrices = kernel * houseSizes + bias
	model.add(tf.layers.dense({ inputShape: [1], units: 1, activation: 'linear' }));

	// define loss function and optimizer
	// meanAbsoluteError = average( absolute(modelOutput - targets) )
	model.compile({
		optimizer: tf.train.sgd(0.001),
		loss: 'meanAbsoluteError'
	});

	// setup training visulisation
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	const container = {
		name: 'Model Training',
		styles: {
			height: '1000px'
		}
	};
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

	// training the model
	await model.fit(
		train_tensors.houseSizes,
		train_tensors.housePrices,
		{
			validationData: [test_tensors.houseSizes, test_tensors.housePrices],
			epochs: 20,
			shuffle: true,
			callbacks: fitCallbacks
		}
	);

	// test the model
	const evalOutput = await model.evaluate(test_tensors.houseSizes, test_tensors.housePrices).data();
	console.log(
		`Evaluation results:\n` +
		`  Loss = ${evalOutput}`);

	// visulize model
	const plot_predictions = [];
	for(var house_size = 500; house_size <= 4000; house_size+=100){
		const house_price_pred = await model.predict(tf.tensor2d([[house_size]])).data();
		plot_predictions.push({
			x: house_size,
			y: house_price_pred
		});
	}
	tfvis.render.linechart(
		{ name: 'Predictions', tab: 'Charts' },
		{ values: [plot_predictions], series: ['model'] },
		{
			width: 400,
			height: 300,
			xAxisDomain: [500, 4000],
			yAxisDomain: [100000, 450000],
			xLabel: 'Size',
			yLabel: 'Price'
		});
});

$("#single-neuron").click(async function () {
	$("#example-title").text("Single Neuron");
    const num_data_points = 1000;
    const data_points_factor = 4;

    const plot_series1_values = [];
    const plot_series2_values = [];
    for(var i = 0; i < num_data_points; i++) {
        if (i % 2 == 0){
            const x_point = getRndFloatG(-4.0, 2.0, data_points_factor);
            const y_point = getRndFloatG(-6.0, 1.0, data_points_factor);
            plot_series1_values.push({ x: x_point, y: y_point });
        } else {
            const x_point = getRndFloatG(-3.0, 3.0, data_points_factor);
            const y_point = getRndFloatG(-12.0, -5.0, data_points_factor);
            plot_series2_values.push({ x: x_point, y: y_point });
        }
    }

    tfvis.render.scatterplot(
        { name: 'Single Neuron Data Points', tab: 'Charts' },
        { values: [plot_series1_values, plot_series2_values], series: ['series 1', 'series 2'] },
        {
            width: 400
        }
    );
});

$("#deep-circles").click(async function () {
	$("#example-title").text("Deep Circles");
    const num_data_points = 1000;
    const data_points_factor = 2;

    const plot_series1_values = [];
    const plot_series2_values = [];
    const num_series_data_point = Math.floor(num_data_points / 2);
    var i = 0;
    while(i < num_data_points) {
        if (i % 2 == 0){
            if (plot_series1_values.length > num_series_data_point) continue;
            const x_point = getRndFloatG(-1.7, 1.7, data_points_factor);
            const y_point = getRndFloatG(-1.7, 1.7, data_points_factor);
            if (x_point < -0.8 || x_point > 0.8 || y_point < -0.8 || y_point > 0.8) {
                plot_series1_values.push({ x: x_point, y: y_point });
                i++;
            }
        } else {
            if (plot_series2_values.length > num_series_data_point) continue;
            const x_point = getRndFloatG(-1.0, 1.0, data_points_factor);
            const y_point = getRndFloatG(-1.0, 1.0, data_points_factor);
            plot_series2_values.push({ x: x_point, y: y_point });
            i++;
        }
    }

    tfvis.render.scatterplot(
        { name: 'Deep Circles Data Points', tab: 'Charts' },
        { values: [plot_series1_values, plot_series2_values], series: ['series 1', 'series 2'] },
        {
            width: 400
        }
    );
});

$( document ).ready(async function () {
});
