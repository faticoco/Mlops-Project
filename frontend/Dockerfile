FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the code
COPY . .

# Expose the port
EXPOSE 5173

# Use the --host flag to bind to all interfaces
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]