# Build stage
FROM node:18 AS build

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install 

# Copy project files
COPY . .

# Build the app
RUN npm run build

# Use the same image for serving
FROM node:18

WORKDIR /app

# Install serve globally
RUN npm install -g serve

# Copy built files from build stage
COPY --from=build /app/dist ./dist

# Expose port
EXPOSE 3000

# Start serve
CMD ["serve", "-s", "dist", "-l", "3000"]