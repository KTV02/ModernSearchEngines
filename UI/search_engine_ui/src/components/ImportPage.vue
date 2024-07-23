<template>
  <div class="import-page">
    <div v-if="isLoading" class="loading-screen">
      <p>Loading...</p>
    </div>
    <div v-else class="content-wrapper">
      <h1>Upload Text File</h1>
      <input ref="fileInput" type="file" @change="handleFileUpload" accept=".txt" />
      <div v-if="downloadUrl" class="download-link">
        <a :href="downloadUrl" download="processed_file.txt">Download Processed File</a>
      </div>
      <div v-if="results.length" class="results">
        <h2>Results</h2>
        <div v-for="(result, index) in results" :key="index" class="result-item">
          <h3>Query {{ result.queryNumber }}</h3>
          <div v-for="(res, i) in result.relevantResults" :key="i" class="result-detail">
            <p>{{ res.rank }}. <a :href="res.url" target="_blank">{{ res.url }}</a> (Score: {{ res.score }})</p>
          </div>
        </div>
      </div>
      <button v-if="downloadUrl" @click="startAgain" class="start-again-button">Start Again</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'ImportPage',
  data() {
    return {
      results: [],
      isLoading: false,
      downloadUrl: null,
      errorMessage: ''
    };
  },
  methods: {
    async handleFileUpload(event) {
      const file = event.target.files[0];
      if (file) {
        this.isLoading = true;
        const formData = new FormData();
        formData.append('file', file);
        try {
          const response = await axios.post('http://localhost:5000/rank_batch', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            responseType: 'blob' // Expect a blob response for the file
          });
          const blob = new Blob([response.data], { type: 'text/plain' });
          this.downloadUrl = URL.createObjectURL(blob);
          const resultsText = await blob.text();
          this.results = this.processResults(resultsText);
        } catch (error) {
          this.errorMessage = 'Error uploading file. Please try again.';
          console.error('Error uploading file:', error);
        } finally {
          this.isLoading = false;
        }
      }
    },
    processResults(text) {
      const results = [];
      const queries = text.trim().split('\n');
      queries.forEach(query => {
        const parts = query.split('\t');
        if (parts.length < 4) {
          console.error(`Skipping malformed line: ${query}`);
          return;
        }
        const [queryNumber, rank, url, scoreStr] = parts;
        let score;
        try {
          score = parseFloat(scoreStr);
        } catch (error) {
          console.error(`Skipping malformed score: ${scoreStr} for query ${query}`);
          return;
        }
        if (isNaN(score)) {
          console.error(`Skipping malformed score: ${scoreStr} for query ${query}`);
          return;
        }
        const existingQuery = results.find(result => result.queryNumber === queryNumber);
        if (existingQuery) {
          existingQuery.relevantResults.push({ rank, url, score });
        } else {
          results.push({
            queryNumber,
            relevantResults: [{ rank, url, score }]
          });
        }
      });
      return results;
    },
    startAgain() {
      this.results = [];
      this.downloadUrl = null;
      this.isLoading = false;
      this.errorMessage = '';
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = null;
      }
    }
  }
};
</script>

<style scoped>
.import-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #f5f5f5; /* Light background color for the page */
}

.loading-screen {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  font-size: 1.5em;
  color: #333;
}

.content-wrapper {
  background-color: white;
  padding: 40px;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  width: 80%;
  max-width: 800px;
  text-align: center;
  color: #333; /* Ensure text color is visible */
}

h1, h2, h3 {
  color: #333; /* Ensure heading text color is visible */
}

input[type="file"] {
  margin: 20px 0;
  color: #333; /* Ensure file input text color is visible */
}

.download-link a {
  color: #007bff; /* Ensure link color is visible */
}

.download-link a:hover {
  color: #0056b3;
}

.results {
  margin-top: 20px;
  color: #333; /* Ensure results text color is visible */
}

.result-item {
  margin-bottom: 20px;
}

.start-again-button {
  margin-top: 20px;
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.start-again-button:hover {
  background-color: #0056b3;
}
</style>
