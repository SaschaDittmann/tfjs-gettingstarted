$("#simple-neural-network").click(async function () {
	$("#example-title").text("Deep Circles");
    $("#output-log").empty();
    const num_data_points = 1000;
    const data_points_factor = 2;

    $("#output-log").append(`<li>Generating demo data</li>`);
    const plot_series1_values = [];
	const plot_series2_values = [];
	const train_x = [];
	const train_y = [];
	const test_x = [];
	const test_y = [];
    const num_series_data_point = Math.floor(num_data_points / 2);
    const num_train_samples = Math.floor(num_data_points * 0.7)
    var i = 0;
    while(i < num_data_points) {
		if (i % 2 == 0){
            if (plot_series1_values.length > num_series_data_point) continue;
            const x_point = getRndFloatG(-1.7, 1.7, data_points_factor);
            const y_point = getRndFloatG(-1.7, 1.7, data_points_factor);
            if (x_point < -0.8 || x_point > 0.8 || y_point < -0.8 || y_point > 0.8) {
                plot_series1_values.push({ x: x_point, y: y_point });
				if (i < num_train_samples) {
					train_x.push([x_point, y_point]);
					train_y.push(i % 2);
				} else {
					test_x.push([x_point, y_point]);
					test_y.push(i % 2);
				}
				i++;
            }
        } else {
            if (plot_series2_values.length > num_series_data_point) continue;
            const x_point = getRndFloatG(-1.0, 1.0, data_points_factor);
            const y_point = getRndFloatG(-1.0, 1.0, data_points_factor);
            plot_series2_values.push({ x: x_point, y: y_point });
			if (i < num_train_samples) {
				train_x.push([x_point, y_point]);
				train_y.push(i % 2);
			} else {
				test_x.push([x_point, y_point]);
				test_y.push(i % 2);
			}
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

	$("#output-log").append(`<li>Converting demo data to tensors</li>`);
	const train_tensors = {
		values: tf.tensor2d(train_x, [train_x.length, 2]),
		labels: tf.tensor2d(train_y, [train_y.length, 1])
	};
	const test_tensors = {
		values: tf.tensor2d(test_x, [test_x.length, 2]),
		labels: tf.tensor2d(test_y, [test_y.length, 1])
	};

	// define model structure
	$("#output-log").append(`<li>Defining model structure</li>`);
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [2], units: 4, activation: 'tanh' }));
	model.add(tf.layers.dense({ units: 4, activation: 'tanh' }));
	model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

	// define loss function and optimizer
	$("#output-log").append(`<li>Defining loss function and optimizer</li>`);
	model.compile({
		optimizer: tf.train.adam(0.05),
		loss: 'binaryCrossentropy',
		metrics: ['acc']
	});

	// setup training visualisation
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	const container = {
		name: 'Model Training',
		styles: {
			height: '1000px'
		}
	};
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

	// training the model
	$("#output-log").append(`<li>Training the model</li>`);
	const history = await model.fit(
		train_tensors.values,
		train_tensors.labels,
		{
			validationData: [test_tensors.values, test_tensors.labels],
			epochs: 20,
			shuffle: true,
			callbacks: [
				fitCallbacks
			]
		}
	);

	// evaluate the model
	$("#output-log").append(`<li>Evaluating the model</li>`);
	const evalOutput = await await model.evaluate(test_tensors.values, test_tensors.labels);
	console.log(
		`Evaluation results:\n` +
		`  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
		`Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
	$("#output-log").append(`<li>Evaluation results:<br/>`+
	`Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}<br/>`+
	`Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`+
	`</li>`);
	
	$("#output-log").append(`<li>Completed demo successfully</li>`);
});
