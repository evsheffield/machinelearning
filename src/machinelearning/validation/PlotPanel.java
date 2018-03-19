package machinelearning.validation;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 * An abstract representation of a GRAL plot which provides some base
 * defaults as well as a mechanism for displaying the created plot
 * in a JFrame.
 *
 * Based on examples provided in the GRAL example repository:
 * https://github.com/eseifert/gral/blob/master/gral-examples/src/main/java/de/erichseifert/gral/examples/ExamplePanel.java
 *
 * @author evanc
 */
public abstract class PlotPanel extends JPanel {
	protected static final Color COLOR1 = new Color(55, 170, 200);
	protected static final Color COLOR2 = new Color(200, 80, 75);

	public PlotPanel() {
		super(new BorderLayout());
		setPreferredSize(new Dimension(1600, 1200));
		setBackground(Color.WHITE);
	}

	public abstract String getTitle();

	public abstract String getDescription();

	/**
	 * Displays the plot in a JFrame
	 *
	 * @return The frame with the plot rendered in it
	 */
	public JFrame showInFrame() {
		JFrame frame = new JFrame(getTitle());
		frame.getContentPane().add(this, BorderLayout.CENTER);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(getPreferredSize());
		frame.setVisible(true);
		return frame;
	}

	@Override
	public String toString() {
		return getTitle();
	}
}
