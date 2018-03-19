package machinelearning.exception;

/**
 * An exception that can be thrown to indicate that
 * the required functionality has not yet been implemented.
 *
 * @author evanc
 */
public class NotImplementedException extends Exception {
	private static final long serialVersionUID = 6093455983900304963L;
	public NotImplementedException() {}
	public NotImplementedException(String s) {
		super(s);
	}
}
