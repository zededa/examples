# Use OpenVINO Model Server as base image
FROM openvino/model_server:latest-gpu

# Switch to root to create directories and set permissions
USER root

# Create models directory for volume mounting
# The actual model files will be mounted from external volume at runtime
RUN mkdir -p /models && chown 5000:5000 /models && chmod 755 /models

# Create a placeholder models directory for fallback (if needed)
RUN mkdir -p /models-fallback && chown 5000:5000 /models-fallback && chmod 755 /models-fallback

# Create entrypoint script to handle volume mounting
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch back to the OpenVINO model server user for security
USER 5000:5000

# Expose the required ports
EXPOSE 8000 9000

# Use custom entrypoint that checks for mounted models
ENTRYPOINT ["/entrypoint.sh"]

# Default arguments for the model server (config-based by default)
CMD ["--config_path", "/models/config.json", \
     "--port", "9000", \
     "--rest_port", "8000", \
     "--log_level", "DEBUG"]
