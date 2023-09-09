% Generate 100 equally spaced values of x
x = linspace(0, 2, 100);

% Calculate y values for the entire range
y = sin(x.^3 - 2);

% Randomly select 10 x values
random_indices = randperm(100, 10);
selected_x = x(random_indices);
selected_y = y(random_indices);

% Create a plot with a solid black line
plot(x, y, 'k-'); 

hold on; 

% Plot the randomly selected points in red
plot(selected_x, selected_y, 'ro'); 

hold off; 

% Add labels and title
xlabel('x');
ylabel('y');
title('Plot of y = sin(x^3 - 2) with Randomly Selected Points');

% Show a legend
legend('y = sin(x^3 - 2)', 'Randomly Selected Points');

% Display the graph
grid on; 
