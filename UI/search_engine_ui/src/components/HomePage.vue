<template>
  <div>
    <div class="search-container">
      <SearchComponent />
    </div>
    <footer class="footer">
      <p class="footer-message">Thank you for visiting our site!</p>
      <button type="button" class="import-button" @click="goToImportPage">Import</button>
    </footer>
  </div>
</template>

<script>
import SearchComponent from './SearchComponent.vue';

export default {
  name: 'HomePage',
  components: {
    SearchComponent
  },
  methods: {
    goToImportPage() {
      this.$router.push({ name: 'import' });
    }
  }
};
</script>

<style>
body {
  margin: 0;
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 100vh;
  text-align: center;
  color: #ffffff;
  background-image: url('@/assets/background.webp');
  background-size: cover;
  background-position: center -170px;
  background-repeat: no-repeat;
  background-attachment: fixed;
}

.search-container {
  margin-top: 85px;
}

.footer {
  background: #444444; /* Changed to dark grey */
  padding: 10px 0;
  width: 100%;
  position: fixed;
  bottom: 0;
  left: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.footer-message {
  margin: 0;
  padding: 5px 0;
  font-size: 1rem;
  color: #ffffff;
}

.import-button {
  background: transparent;
  border: 1px solid white;
  color: white;
  padding: 10px 20px;
  cursor: pointer;
  border-radius: 4px;
  margin-top: 5px;
}

.import-button:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

/* Additional styles for SearchComponent (you may need to adjust these in the SearchComponent.vue file) */
:deep(.search-bar) {
  background-color: black;
  color: white;
}

:deep(.search-engine-logo) {
  filter: invert(1);
}

:deep(.university-logo) {
  filter: invert(1);
  background-color: #444444; /* Changed to dark grey */
}

:deep(.search-button) {
  background-color: #00008B; /* Darker blue */
  color: white;
}

:deep(.display-decision-tree-button) {
  background-color: #006400; /* Darker green */
  color: white;
}
</style>