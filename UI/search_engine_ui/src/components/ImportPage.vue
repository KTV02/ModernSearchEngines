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
          <h3>{{ result.query }}</h3>
          <div v-for="(title, i) in result.relevantTitles" :key="i">
            <a :href="result.relevantUrls[i]" target="_blank">{{ title }}</a>
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
      downloadUrl: null
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
          console.error('Error uploading file:', error);
        } finally {
          this.isLoading = false;
        }
      }
    },
    processResults(text) {
      const results = [];
      const queries = text.split('\n\n'); // assuming each result is separated by double newlines
      queries.forEach(query => {
        const lines = query.split('\n');
        if (lines.length > 1) {
          const queryLine = lines[0].replace('Query: ', '');
          const resultsLine = lines[1].replace('Results: ', '');
          const relevantTitles = resultsLine.split(', ');
          results.push({
            query: queryLine,
            relevantTitles: relevantTitles,
            relevantUrls: relevantTitles.map(title => `http://example.com/${title.replace(' ', '_')}`) // Dummy URL
          });
        }
      });
      return results;
    },
    startAgain() {
      this.results = [];
      this.downloadUrl = null;
      this.isLoading = false;
      // Reset the file input value
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
}

input[type="file"] {
  margin: 20px 0;
}

.download-link {
  margin: 20px 0;
}

.results {
  margin-top: 20px;
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
