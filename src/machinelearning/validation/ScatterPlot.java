package machinelearning.validation;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;
import de.erichseifert.gral.ui.InteractivePanel;
import de.erichseifert.gral.util.GraphicsUtils;

/**
 * A scatter plot for showing one or more series of data
 * using GRAL.
 *
 * @author evanc
 */
public class ScatterPlot extends PlotPanel {

	public String name;
	public String description;

	/**
	 * Initializes a new scatter plot.
	 *
	 * @param name The short name of the graph to use in the panel title.
	 * @param description The full graph description to display above the graph.
	 * @param data Rows of data to represent in the graph
	 * @param seriesNames The names of the series to use in the legend
	 * @param xAxisLabel The label to apply to the x-axis.
	 * @param yAxisLabel The label to apply to the y-axis.
	 */
	public ScatterPlot(String name, String description, ArrayList<ArrayList<Double>> data, ArrayList<String> seriesNames,
			String xAxisLabel, String yAxisLabel) {
		this.name = name;
		this.description = description;

		// Import the data
		DataTable dataTable = new DataTable(data.get(0).size(), Double.class);
		for(ArrayList<Double> row : data) {
			dataTable.add(row);
		}

		// Create a series from the data. The first value in a row is interpreted as the x-value.
		// All subsequent values are the y value of the various series. E.g. for data row 1,2,3,
		// 1 is the x value, and 2 is the y-value at x=1 for series 1, and 3 is the y-value at x=1
		// for series two.
		ArrayList<DataSeries> series = new ArrayList<DataSeries>();
		for(int i = 1; i < dataTable.getColumnCount(); i++) {
			DataSeries s = new DataSeries(seriesNames.get(i - 1), dataTable, 0, i);
			series.add(s);
		}

		// Create the plot
		XYPlot plot = new XYPlot(series.toArray(new DataSeries[series.size()]));

		// Define various fonts
		Font titleFont = new Font("Default", Font.BOLD, 42);
		Font font = new Font("Default", Font.PLAIN, 24);
		Font axisTickFont = new Font("Default", Font.PLAIN, 20);

		// Apply the graph title
		plot.getTitle().setText(getDescription());
		plot.getTitle().setFont(titleFont);

		// Set the renderers for the various series
		Color currentColor = COLOR1;
		int seriesCount = 0;
		for(DataSeries s : series) {
			seriesCount++;
			if(seriesCount == 4) {
				currentColor = COLOR2;
			}
			PointRenderer defaultRenderer = new DefaultPointRenderer2D();
			Shape oldShape = defaultRenderer.getShape();
			Shape newShape = AffineTransform.getScaleInstance(2.5, 2.5).createTransformedShape(oldShape);
			defaultRenderer.setShape(newShape);
			defaultRenderer.setColor(currentColor);
			plot.setPointRenderers(s, defaultRenderer);
			currentColor = GraphicsUtils.deriveDarker(currentColor);

//			DiscreteLineRenderer2D discreteRenderer = new DiscreteLineRenderer2D();
//			discreteRenderer.setColor(currentColor);
//			discreteRenderer.setStroke(new BasicStroke(2.0f, BasicStroke.CAP_BUTT,
//				    BasicStroke.JOIN_MITER, 10.0f, new float[] {3f, 3f}, 0.0f));
			LineRenderer lineRend = new DefaultLineRenderer2D();
			lineRend.setColor(GraphicsUtils.deriveWithAlpha(currentColor, 128));

			plot.setLineRenderers(s, lineRend);
		}

		// Pad the edges of the graph so that the axis titles will display properly
		double insetsTop = 20.0,
			       insetsLeft = 100.0,
			       insetsBottom = 100.0,
			       insetsRight = 40.0;
		plot.setInsets(new Insets2D.Double(
		    insetsTop, insetsLeft, insetsBottom, insetsRight));

		// Set labels for the axes
		plot.getAxisRenderer(XYPlot.AXIS_X).getLabel().setText(xAxisLabel);
		plot.getAxisRenderer(XYPlot.AXIS_X).getLabel().setFont(font);
		plot.getAxisRenderer(XYPlot.AXIS_X).setTickFont(axisTickFont);
		plot.getAxisRenderer(XYPlot.AXIS_Y).getLabel().setText(yAxisLabel);
		plot.getAxisRenderer(XYPlot.AXIS_Y).getLabel().setRotation(90);
		plot.getAxisRenderer(XYPlot.AXIS_Y).getLabel().setFont(font);
		plot.getAxisRenderer(XYPlot.AXIS_Y).setTickFont(axisTickFont);

		// Create the legend
		plot.setLegendVisible(true);
		plot.getLegend().setAlignmentX(0.9);
		plot.getLegend().setAlignmentY(0.1);
		plot.getLegend().setFont(font);

		add(new InteractivePanel(plot), BorderLayout.CENTER);
	}

	@Override
	public String getDescription() {
		return description;
	}

	@Override
	public String getTitle() {
		return name;
	}
}
