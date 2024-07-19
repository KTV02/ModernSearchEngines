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
  color: #2c3e50;
  background-image: url('@/assets/background.webp'); /* Adjust the path to your image */
  background-size: cover;
  background-position: center -170px; /* Move the background image further up */
  background-repeat: no-repeat;
  background-attachment: fixed; /* Ensure the background image stays fixed during scrolling */
}

.search-container {
  margin-top: 85px; /* Move the search component down by 85 pixels */
}

.footer {
  background: rgba(255, 255, 255, 0.8);
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
  color: #2c3e50;
}

.import-button {
  background: transparent;
  border: 1px solid black;
  padding: 10px 20px;
  cursor: pointer;
  border-radius: 4px;
  margin-top: 5px;
}

.import-button:hover {
  background-color: rgba(0, 0, 0, 0.1);
}
</style>
