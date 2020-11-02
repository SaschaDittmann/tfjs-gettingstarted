$("#simple-regression-button").click(async function () {
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

	// Normalise Data
	const train_house_sizes_norm = normalize(train_house_sizes);
	const train_house_prices_norm = normalize(train_house_prices);
	const test_house_sizes_norm = normalize(test_house_sizes);
	const test_house_prices_norm = normalize(test_house_prices);

	const train_tensors = {
		houseSizes: tf.tensor2d(train_house_sizes, [train_house_sizes.length, 1]),
		housePrices: tf.tensor2d(train_house_prices, [train_house_prices.length, 1])
	};
	const test_tensors = {
		houseSizes: tf.tensor2d(test_house_sizes, [test_house_sizes.length, 1]),
		housePrices: tf.tensor2d(test_house_prices, [test_house_prices.length, 1])
	};
	
	const model = tf.sequential();

	// housePrices = kernel * houseSizes + bias
	model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
	
	// meanAbsoluteError = average( absolute(modelOutput - targets) )
	model.compile({
		optimizer: tf.train.sgd(0.001), 
		loss: 'meanAbsoluteError'
	});

	// Setup trainings visulisation
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc']
	const container = { 
		name: 'Model Training', 
		styles: { 
			 height: '1000px' 
		}};
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

	await model.fit(
		train_tensors.houseSizes, 
		train_tensors.housePrices, 
		{
			validationData: [test_tensors.houseSizes, test_tensors.housePrices],
			epochs: 20,
			shuffle: true,
			callbacks: fitCallbacks
		});
	
	const evalOutput = await model.evaluate(test_tensors.houseSizes, test_tensors.housePrices).data();
    console.log(
        `Evaluation results:\n` +
		`  Loss = ${evalOutput}`);
	
	const plot_predictions = [];
	for	(var house_size=500; house_size<=4000; house_size+=100){
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

$( document ).ready(async function () {
});
