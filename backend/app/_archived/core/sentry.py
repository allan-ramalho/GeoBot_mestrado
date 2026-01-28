"""
Sentry Integration for Error Tracking
Captures and reports crashes and errors
"""

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import logging
from app.core.config import settings

def init_sentry():
    """Initialize Sentry SDK"""
    
    if not settings.SENTRY_DSN:
        logging.info("Sentry DSN not configured, skipping Sentry init")
        return
    
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        release=f"geobot@{settings.VERSION}",
        
        # Integrations
        integrations=[
            FastApiIntegration(
                transaction_style="endpoint",
                failed_request_status_codes=[400, 500],
            ),
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            ),
        ],
        
        # Performance monitoring
        traces_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
        
        # Error sampling
        sample_rate=1.0,
        
        # Before send hook
        before_send=before_send_hook,
        
        # Attach stack traces
        attach_stacktrace=True,
        
        # Max breadcrumbs
        max_breadcrumbs=50,
        
        # Debug
        debug=settings.ENVIRONMENT == "development",
    )
    
    logging.info(f"Sentry initialized (env={settings.ENVIRONMENT}, release={settings.VERSION})")


def before_send_hook(event, hint):
    """
    Filter events before sending to Sentry
    Use this to scrub sensitive data or filter certain errors
    """
    
    # Don't send API key validation errors (expected behavior)
    if 'exception' in event:
        for exception in event['exception']['values']:
            if 'Invalid API key' in exception.get('value', ''):
                return None
    
    # Scrub API keys from event
    if 'request' in event:
        if 'headers' in event['request']:
            headers = event['request']['headers']
            for key in ['Authorization', 'X-Api-Key', 'Api-Key']:
                if key in headers:
                    headers[key] = '[Filtered]'
    
    # Add custom context
    event.setdefault('contexts', {})
    event['contexts']['app'] = {
        'version': settings.VERSION,
        'environment': settings.ENVIRONMENT,
    }
    
    return event


def capture_exception(exception: Exception, **kwargs):
    """
    Manually capture exception with additional context
    
    Usage:
        try:
            risky_operation()
        except Exception as e:
            capture_exception(e, level='error', extra={'user_id': 123})
    """
    sentry_sdk.capture_exception(exception, **kwargs)


def capture_message(message: str, level: str = 'info', **kwargs):
    """
    Capture custom message
    
    Usage:
        capture_message('User completed workflow', level='info', extra={'workflow': 'magnetic_enhancement'})
    """
    sentry_sdk.capture_message(message, level=level, **kwargs)


def set_user_context(user_id: str, email: str = None, username: str = None):
    """
    Set user context for error reports
    
    Usage:
        set_user_context(user_id='12345', email='user@example.com')
    """
    sentry_sdk.set_user({
        'id': user_id,
        'email': email,
        'username': username,
    })


def set_custom_context(key: str, value: dict):
    """
    Add custom context to error reports
    
    Usage:
        set_custom_context('processing', {
            'function': 'reduction_to_pole',
            'params': {'inclination': -30},
            'grid_size': '1000x1000'
        })
    """
    sentry_sdk.set_context(key, value)


def add_breadcrumb(message: str, category: str = 'default', level: str = 'info', data: dict = None):
    """
    Add breadcrumb for debugging context
    
    Usage:
        add_breadcrumb('User started processing', category='processing', data={'function': 'rtp'})
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


# Decorator for automatic error tracking
def track_errors(func):
    """
    Decorator to automatically track errors in functions
    
    Usage:
        @track_errors
        def risky_function():
            ...
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            capture_exception(e, extra={
                'function': func.__name__,
                'args': str(args)[:100],  # Limit length
                'kwargs': str(kwargs)[:100],
            })
            raise
    return wrapper


# Example usage in processing functions
def example_processing_with_sentry():
    """Example of Sentry integration in processing"""
    
    # Add breadcrumb
    add_breadcrumb('Starting magnetic processing', category='processing')
    
    # Set context
    set_custom_context('processing', {
        'function': 'reduction_to_pole',
        'grid_size': '1000x1000',
    })
    
    try:
        # Your processing code
        result = perform_rtp()
        
        # Success breadcrumb
        add_breadcrumb('Processing completed', category='processing', level='info')
        
        return result
        
    except Exception as e:
        # Add error context
        add_breadcrumb('Processing failed', category='processing', level='error')
        
        # Capture with extra context
        capture_exception(e, extra={
            'stage': 'rtp_execution',
            'last_successful_step': 'data_validation',
        })
        
        raise


def perform_rtp():
    """Dummy function for example"""
    pass
